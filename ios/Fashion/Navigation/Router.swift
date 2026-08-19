import SwiftUI

enum Tab: Hashable {
    case closet, search, feed
}

/// One destination type per navigable thing, so `NavigationStack` can restore
/// and deep-link into any of them.
enum Destination: Hashable {
    case outfit(OutfitResult)
    case closetItem(UUID)
}

extension OutfitResult: Hashable {
    static func == (lhs: OutfitResult, rhs: OutfitResult) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }
}

/// Navigation state, including deep links.
///
/// Deep links exist for Siri: a spoken query returns results, and tapping one
/// has to land on that exact outfit. There is no "fetch outfit by id"
/// endpoint, so the intent writes what it found into `SpokenResultsCache` and
/// the link resolves against that. When the cache misses — the app was
/// reinstalled, the entry aged out — the link degrades to re-running the
/// original text query rather than showing a dead screen. That fallback is
/// the whole reason the query text travels in the URL.
@MainActor
@Observable
final class Router {
    var selectedTab: Tab = .search
    var closetPath = NavigationPath()
    var searchPath = NavigationPath()
    var feedPath = NavigationPath()

    /// Set by a deep link; consumed by `SearchView` on appear.
    var pendingSearchText: String?

    func open(_ destination: Destination, in tab: Tab) {
        selectedTab = tab
        switch tab {
        case .closet: closetPath.append(destination)
        case .search: searchPath.append(destination)
        case .feed: feedPath.append(destination)
        }
    }

    /// `fashion://outfit/<id>?q=<original query>`
    /// `fashion://search?q=<text>`
    func handle(_ url: URL) {
        guard url.scheme == "fashion" else { return }
        let components = URLComponents(url: url, resolvingAgainstBaseURL: false)
        let query = components?.queryItems?.first { $0.name == "q" }?.value

        switch url.host {
        case "outfit":
            let id = url.pathComponents.filter { $0 != "/" }.first ?? ""
            if let cached = SpokenResultsCache.shared.outfit(id: id) {
                selectedTab = .search
                searchPath = NavigationPath()
                searchPath.append(Destination.outfit(cached))
                pendingSearchText = query
            } else {
                // Honest degradation: we cannot show an outfit we no longer
                // hold, so re-ask the question that produced it.
                selectedTab = .search
                searchPath = NavigationPath()
                pendingSearchText = query
            }
        case "search":
            selectedTab = .search
            searchPath = NavigationPath()
            pendingSearchText = query
        default:
            break
        }
    }
}
