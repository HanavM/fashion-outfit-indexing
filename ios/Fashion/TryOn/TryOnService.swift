import Foundation
import UIKit

/// The result of "See it on you".
///
/// The disclosure string is stored *on the result*, not applied by whichever
/// view happens to render it. Anything that displays or writes this image is
/// therefore holding the words "generated, not a photograph" in the same
/// value, and cannot present the picture without them.
struct TryOnPreview: Sendable {
    let image: UIImage
    let generatedAt: Date
    let sourceOutfitID: String

    /// Non-negotiable label. Deliberately a constant with no setter.
    let disclosure = "Generated image — not a photograph of you or of this outfit."
}

protocol TryOnService: Sendable {
    /// False when no endpoint has been configured. The UI uses this to hide
    /// or disable the feature honestly rather than offering a button that
    /// always fails.
    var isConfigured: Bool { get }

    func generatePreview(personImage: Data,
                         outfit: OutfitResult,
                         referenceImage: Data?) async throws -> TryOnPreview
}

enum TryOnError: LocalizedError {
    case notConfigured
    case refusedByModel(String)
    case transport(String)
    case badResponse

    var errorDescription: String? {
        switch self {
        case .notConfigured:
            return "No image endpoint configured"
        case .refusedByModel(let reason):
            return reason
        case .transport(let message):
            return message
        case .badResponse:
            return "The image service replied with something unreadable"
        }
    }
}

/// Azure AI Foundry, behind REST.
///
/// **Unverified.** No endpoint had been provisioned when this was written, so
/// the request and response shapes below follow Azure OpenAI's image API
/// (`/images/generations`, `api-key` header, `data[0].b64_json`) and have not
/// been exercised against a live deployment. The protocol exists precisely so
/// that correcting this is a change to one type. Anything in the app that
/// depends on the *shape* of this call is a bug.
struct AzureTryOnService: TryOnService {
    let endpoint: URL?
    let apiKey: String?
    private let session: URLSession

    init(bundle: Bundle = .main, session: URLSession? = nil) {
        let rawEndpoint = (bundle.object(forInfoDictionaryKey: "AzureTryOnEndpoint") as? String)?.trimmed
        endpoint = (rawEndpoint?.isEmpty == false) ? URL(string: rawEndpoint!) : nil
        let rawKey = (bundle.object(forInfoDictionaryKey: "AzureTryOnAPIKey") as? String)?.trimmed
        apiKey = (rawKey?.isEmpty == false) ? rawKey : nil

        if let session {
            self.session = session
        } else {
            let config = URLSessionConfiguration.default
            config.timeoutIntervalForRequest = 120
            self.session = URLSession(configuration: config)
        }
    }

    var isConfigured: Bool { endpoint != nil && apiKey != nil }

    func generatePreview(personImage: Data,
                         outfit: OutfitResult,
                         referenceImage: Data?) async throws -> TryOnPreview {
        guard let endpoint, let apiKey else { throw TryOnError.notConfigured }

        struct Body: Encodable {
            let prompt: String
            let n: Int
            let size: String
            let image: String
            let reference_image: String?
        }

        let described = outfit.categories.isEmpty
            ? "the outfit in the reference photograph"
            : outfit.categories.joined(separator: ", ")

        let body = Body(
            prompt: """
            Show the person in the supplied photograph wearing \(described). \
            Keep the person's face, body and proportions unchanged. \
            Plain background, natural daylight.
            """,
            n: 1,
            size: "1024x1024",
            image: personImage.base64EncodedString(),
            reference_image: referenceImage?.base64EncodedString())

        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue(apiKey, forHTTPHeaderField: "api-key")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(body)

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            throw TryOnError.transport(error.localizedDescription)
        }

        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            struct Failure: Decodable {
                struct Inner: Decodable { let message: String? }
                let error: Inner?
            }
            let message = (try? JSONDecoder().decode(Failure.self, from: data))?.error?.message
            throw TryOnError.refusedByModel(message ?? "The image service returned \(http.statusCode).")
        }

        struct Success: Decodable {
            struct Item: Decodable { let b64_json: String? }
            let data: [Item]?
        }

        guard let payload = try? JSONDecoder().decode(Success.self, from: data),
              let encoded = payload.data?.first?.b64_json,
              let bytes = Data(base64Encoded: encoded),
              let image = UIImage(data: bytes) else {
            throw TryOnError.badResponse
        }

        return TryOnPreview(image: image, generatedAt: .now, sourceOutfitID: outfit.id)
    }
}

/// Preview/stub implementation.
///
/// It reports itself as configured but produces no picture, because
/// fabricating a plausible "generated" image in a stub is exactly the kind of
/// thing this app has promised not to do. Previews of the try-on screen
/// therefore exercise the labelling, the consent copy and the failure state,
/// which is where the design risk actually is.
struct StubTryOnService: TryOnService {
    var isConfigured: Bool = true
    var error: TryOnError? = .notConfigured

    func generatePreview(personImage: Data,
                         outfit: OutfitResult,
                         referenceImage: Data?) async throws -> TryOnPreview {
        try? await Task.sleep(for: .milliseconds(400))
        throw error ?? TryOnError.notConfigured
    }
}
