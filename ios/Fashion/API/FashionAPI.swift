import Foundation

/// Everything the app can ask the backend for.
///
/// The split exists for three reasons, in order of how much they mattered:
///
/// 1. `/outfit_search` **does not exist yet**. It is specified, agreed and
///    being built in parallel. Coding the screens against a protocol means
///    the day it deploys, one type changes.
/// 2. Every screen has to be previewable in Xcode with no network and no
///    credentials. `StubFashionAPI` returns fixture data shaped exactly like
///    the real responses, including the awkward cases — a wrong top-1, an
///    empty result set — so the states that matter get designed, not just
///    the happy one.
/// 3. Cold starts are ~20 seconds. Being able to develop against the stub
///    keeps that cost off the inner loop.
protocol FashionAPI: Sendable {
    func health() async throws -> HealthResponse

    /// Identify a photographed garment against the catalog. This is what
    /// Closet capture uses.
    func identify(imageData: Data, topK: Int) async throws -> IdentifyResponse

    /// The unified catalog route: image and/or text.
    func query(_ request: QueryRequest) async throws -> QueryResponse

    /// Search real outfit photos. Not yet deployed — see above.
    func outfitSearch(_ request: OutfitSearchRequest) async throws -> OutfitSearchResponse

    /// Resolve an outfit's image reference into something loadable.
    ///
    /// On the protocol rather than in the view because the two
    /// implementations answer it differently: the live service returns
    /// absolute URLs (or corpus-relative paths that need the host bolted on),
    /// while the stub has no images at all and returns nil so views fall back
    /// to a described placeholder.
    func imageURL(for outfit: OutfitResult) -> URL?
}

enum FashionAPIError: LocalizedError, Equatable {
    case notConfigured
    case notDeployed(String)
    case unauthorized
    case server(status: Int, detail: String?)
    case transport(String)
    case decoding(String)

    var errorDescription: String? {
        switch self {
        case .notConfigured:
            return "No API key yet"
        case .notDeployed(let name):
            return "\(name) isn’t live yet"
        case .unauthorized:
            return "The API key was rejected"
        case .server(let status, let detail):
            return detail ?? "The server returned \(status)"
        case .transport(let message):
            return message
        case .decoding:
            return "The server sent something this app didn’t understand"
        }
    }

    /// What the user should do about it, in plain words. Shown under the
    /// title in error states. Nil when there is genuinely nothing to do.
    var recoveryAdvice: String? {
        switch self {
        case .notConfigured:
            return "Add one in Settings and this will start working."
        case .notDeployed:
            return "Outfit search is still being built. The rest of the app works."
        case .unauthorized:
            return "Check the key in Settings against the one in the project’s .env."
        case .server:
            return "Try again in a moment."
        case .transport:
            return "Check your connection and try again."
        case .decoding:
            return "This usually means the API changed shape. Worth reporting."
        }
    }
}
