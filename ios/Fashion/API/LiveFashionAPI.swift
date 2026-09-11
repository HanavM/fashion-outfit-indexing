import Foundation

/// The real client.
///
/// Two things about this backend shape the implementation:
///
/// - **Cold starts run about 20 seconds.** The request timeout is therefore
///   generous (90s) rather than the URLSession default of 60, and callers are
///   expected to describe the wait honestly instead of showing a spinner that
///   reads as a hang. See `LoadingState`.
/// - **There are two hosts.** `/health`, `/identify` and `/query` are the
///   identifier; `/outfit_search` and `/photo` are outfit search, which is a
///   separate Modal app. Every request picks its host from the path, because
///   sending one to the other is a 404, not a slow answer.
///
/// A 404 from `/outfit_search` is still translated into `.notDeployed`, but
/// that is now a genuine signal rather than the permanent state it was while
/// both paths shared the identifier's host.
struct LiveFashionAPI: FashionAPI {
    /// Credentials are copied out of `APIConfiguration` at construction
    /// rather than held by reference. `APIConfiguration` is an `@Observable`
    /// class and therefore not `Sendable`, and this client crosses actor
    /// boundaries constantly. Copying is also the correct semantics: when the
    /// key changes, `AppEnvironment.reloadAPI()` builds a new client, so a
    /// live reference would buy nothing.
    private let apiKey: String?
    let baseURL: URL
    let outfitBaseURL: URL
    private let session: URLSession

    init(configuration: APIConfiguration, session: URLSession? = nil) {
        self.apiKey = configuration.apiKey
        self.baseURL = configuration.baseURL
        self.outfitBaseURL = configuration.outfitBaseURL
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
            return try await send(path: "/outfit_search", method: "POST",
                                  body: request, host: outfitBaseURL)
        } catch FashionAPIError.server(let status, _) where status == 404 || status == 501 {
            throw FashionAPIError.notDeployed("Outfit search")
        }
    }

    /// Three shapes arrive here and only one of them used to work.
    ///
    /// 1. An absolute URL — a corpus hosted somewhere public. Used as-is.
    /// 2. `/photo?path=<rel>` — what the deployed service actually emits.
    ///    Resolved against the outfit host, keeping its query intact. The old
    ///    code fell through to case 3 for this and produced
    ///    `/photo?path=/photo%3Fpath%3D...`, which the server resolved to a
    ///    path that is not a file and answered 404. Every photo, every time.
    /// 3. A bare corpus-relative path — what the local engine emits. Wrapped
    ///    in the photo endpoint.
    func imageURL(for outfit: OutfitResult) -> URL? {
        let reference = outfit.imageURL
        guard !reference.isEmpty else { return nil }

        if let url = URL(string: reference), url.scheme != nil { return url }

        if reference.hasPrefix("/") {
            return URL(string: reference, relativeTo: outfitBaseURL)?.absoluteURL
        }

        var components = URLComponents(url: outfitBaseURL.appendingPathComponent("photo"),
                                       resolvingAgainstBaseURL: false)
        components?.queryItems = [URLQueryItem(name: "path", value: reference)]
        return components?.url
    }

    // MARK: Transport

    /// `host` defaults to the identifier, so only the outfit paths have to
    /// say which service they mean.
    private func send<Body: Encodable, Response: Decodable>(
        path: String, method: String, body: Body?, host: URL? = nil
    ) async throws -> Response {
        guard let apiKey, !apiKey.isEmpty else {
            throw FashionAPIError.notConfigured
        }

        let target = host ?? baseURL
        var request = URLRequest(url: target.appendingPathComponent(path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))))
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
