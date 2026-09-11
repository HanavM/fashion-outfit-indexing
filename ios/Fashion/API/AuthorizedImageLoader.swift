import SwiftUI

/// Loads corpus photographs, with credentials, and caches them.
///
/// `AsyncImage` cannot do this job. The `/photo` endpoint is authenticated —
/// it answers 401 without a `Bearer` header, exactly like every other route on
/// the service — and `AsyncImage` offers no way to attach one. The app
/// therefore rendered "Photo unavailable" on every single tile while the
/// search behind it was working perfectly: real results, real scores, no
/// pictures.
///
/// **The header goes only to our own hosts.** `image_url` is allowed to be an
/// absolute URL pointing anywhere the corpus lives — reddit preview links, for
/// one — and attaching the API key to those would hand the credential to a
/// third party. `isTrusted(_:)` is the whole reason this is not simply "always
/// set the header".
@MainActor
@Observable
final class AuthorizedImageLoader {
    static let shared = AuthorizedImageLoader()

    private var apiKey: String?
    /// Hosts that are allowed to receive the key.
    private var trustedHosts: Set<String> = []

    /// `NSCache` rather than a dictionary so the system can evict under
    /// pressure. A grid of outfit photos is the memory-heaviest screen here.
    private let cache: NSCache<NSURL, UIImage> = {
        let cache = NSCache<NSURL, UIImage>()
        cache.countLimit = 240
        return cache
    }()

    /// In-flight requests, so a photo scrolled past and back does not start a
    /// second download of the same bytes.
    private var inFlight: [URL: Task<UIImage?, Never>] = [:]

    private let session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 45
        config.requestCachePolicy = .returnCacheDataElseLoad
        return URLSession(configuration: config)
    }()

    private init() {}

    /// Called by `AppEnvironment` whenever credentials change, so a key
    /// entered in Settings takes effect without a relaunch.
    func configure(apiKey: String?, hosts: [URL]) {
        self.apiKey = apiKey
        trustedHosts = Set(hosts.compactMap(\.host))
        // A key change invalidates nothing visually, but a *cleared* key
        // should not leave authenticated images on screen.
        if apiKey == nil { cache.removeAllObjects() }
    }

    func cached(_ url: URL) -> UIImage? { cache.object(forKey: url as NSURL) }

    private func isTrusted(_ url: URL) -> Bool {
        guard let host = url.host else { return false }
        return trustedHosts.contains(host)
    }

    func image(for url: URL) async -> UIImage? {
        if let hit = cached(url) { return hit }
        if let existing = inFlight[url] { return await existing.value }

        let task = Task<UIImage?, Never> { [session, apiKey] in
            var request = URLRequest(url: url)
            if let apiKey, !apiKey.isEmpty, self.isTrusted(url) {
                request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
            }
            guard let (data, response) = try? await session.data(for: request) else { return nil }
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode),
                  let image = UIImage(data: data) else { return nil }
            return image
        }

        inFlight[url] = task
        let image = await task.value
        inFlight[url] = nil
        if let image { cache.setObject(image, forKey: url as NSURL) }
        return image
    }
}

/// The drop-in replacement for `AsyncImage`'s phase API, so `OutfitPhoto` keeps
/// its three states and its honest empty state rather than growing loading
/// logic of its own.
struct AuthorizedImage<Content: View>: View {
    let url: URL?
    var animation: Animation?
    @ViewBuilder let content: (AsyncImagePhase) -> Content

    @State private var phase: AsyncImagePhase = .empty

    var body: some View {
        content(phase)
            .task(id: url) { await load() }
    }

    private func load() async {
        guard let url else {
            phase = .empty
            return
        }
        // A cache hit must not flash the placeholder on the way past, and must
        // not animate — it was already there.
        if let hit = AuthorizedImageLoader.shared.cached(url) {
            phase = .success(Image(uiImage: hit))
            return
        }
        phase = .empty

        let loaded = await AuthorizedImageLoader.shared.image(for: url)
        guard !Task.isCancelled else { return }
        withAnimation(animation) {
            if let loaded {
                phase = .success(Image(uiImage: loaded))
            } else {
                phase = .failure(URLError(.resourceUnavailable))
            }
        }
    }
}
