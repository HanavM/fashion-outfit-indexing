import Foundation

/// The real client.
///
/// Two things about this backend shape the implementation:
///
/// - **Cold starts run about 20 seconds.** The request timeout is therefore
///   generous (90s) rather than the URLSession default of 60, and callers are
///   expected to describe the wait honestly instead of showing a spinner that
///   reads as a hang. See `LoadingState`.
/// - **`/outfit_search` is not deployed.** A 404 from that path is translated
///   into `.notDeployed` rather than a generic server error, so the UI can
///   say the true thing: this part is still being built.
struct LiveFashionAPI: FashionAPI {
    /// Credentials are copied out of `APIConfiguration` at construction
    /// rather than held by reference. `APIConfiguration` is an `@Observable`
    /// class and therefore not `Sendable`, and this client crosses actor
    /// boundaries constantly. Copying is also the correct semantics: when the
    /// key changes, `AppEnvironment.reloadAPI()` builds a new client, so a
    /// live reference would buy nothing.
    private let apiKey: String?
    let baseURL: URL
    private let session: URLSession

    init(configuration: APIConfiguration, session: URLSession? = nil) {
        self.apiKey = configuration.apiKey
        self.baseURL = configuration.baseURL
        if let session {
            self.session = session
        } else {
            let config = URLSessionConfiguration.default
            config.timeoutIntervalForRequest = 90
            config.timeoutIntervalForResource = 180
            config.waitsForConnectivity = true
            self.session = URLSession(configuration: config)
        }
    }

    private static let decoder = JSONDecoder()
    private static let encoder = JSONEncoder()

    // MARK: Endpoints

    func health() async throws -> HealthResponse {
        try await send(path: "/health", method: "GET", body: Optional<QueryRequest>.none)
    }

    func identify(imageData: Data, topK: Int) async throws -> IdentifyResponse {
        struct Body: Encodable {
            let image_base64: String
            let top_k: Int
        }
        return try await send(path: "/identify", method: "POST",
                              body: Body(image_base64: imageData.base64EncodedString(),
                                         top_k: topK))
    }

    func query(_ request: QueryRequest) async throws -> QueryResponse {
        try await send(path: "/query", method: "POST", body: request)
    }

    func outfitSearch(_ request: OutfitSearchRequest) async throws -> OutfitSearchResponse {
        do {
            return try await send(path: "/outfit_search", method: "POST", body: request)
        } catch FashionAPIError.server(let status, _) where status == 404 || status == 501 {
            throw FashionAPIError.notDeployed("Outfit search")
        }
    }

    func imageURL(for outfit: OutfitResult) -> URL? {
        let reference = outfit.imageURL
        guard !reference.isEmpty else { return nil }

        // Already absolute: the deployed service is expected to hand back
        // full URLs to wherever the corpus is hosted.
        if let url = URL(string: reference), url.scheme != nil { return url }

        // Otherwise it is a corpus-relative path, which is what the local
        // engine emits. Route it through the service's photo endpoint.
        var components = URLComponents(url: baseURL.appendingPathComponent("photo"),
                                       resolvingAgainstBaseURL: false)
        components?.queryItems = [URLQueryItem(name: "path", value: reference)]
        return components?.url
    }

    // MARK: Transport

    private func send<Body: Encodable, Response: Decodable>(
        path: String, method: String, body: Body?
    ) async throws -> Response {
        guard let apiKey, !apiKey.isEmpty else {
            throw FashionAPIError.notConfigured
        }

        var request = URLRequest(url: baseURL.appendingPathComponent(path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))))
        request.httpMethod = method
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        if let body {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try LiveFashionAPI.encoder.encode(body)
        }

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch let error as URLError where error.code == .timedOut {
            throw FashionAPIError.transport(
                "The server didn’t answer in 90 seconds. It may still be starting up.")
        } catch {
            throw FashionAPIError.transport(error.localizedDescription)
        }

        guard let http = response as? HTTPURLResponse else {
            throw FashionAPIError.transport("No response from the server.")
        }

        guard (200..<300).contains(http.statusCode) else {
            if http.statusCode == 401 || http.statusCode == 403 {
                throw FashionAPIError.unauthorized
            }
            throw FashionAPIError.server(status: http.statusCode, detail: Self.detail(from: data))
        }

        do {
            return try LiveFashionAPI.decoder.decode(Response.self, from: data)
        } catch {
            throw FashionAPIError.decoding(String(describing: error))
        }
    }

    /// FastAPI puts its human-readable error under `detail`.
    private static func detail(from data: Data) -> String? {
        struct Detail: Decodable { let detail: String? }
        return (try? decoder.decode(Detail.self, from: data))?.detail
    }
}
