import AppIntents
import SwiftUI

/// The Siri surface.
///
/// App Intents rather than SiriKit: SiriKit's donation-and-domain model has
/// no vocabulary for "search outfits", and App Intents lets the same code
/// answer by voice, appear in Spotlight, and be dropped into a Shortcut.
///
/// The intent speaks a short answer and shows a snippet of the top results.
/// Each result in that snippet is a `Link` into `fashion://outfit/<id>`, and
/// the results are parked in `SpokenResultsCache` first so that link has
/// something to resolve against — there is no "get one outfit by id"
/// endpoint. The original query rides along in the URL so that a cache miss
/// degrades to re-running the search rather than to a dead end.
struct OutfitSearchIntent: AppIntent {
    static var title: LocalizedStringResource = "Search outfits"
    static var description = IntentDescription(
        "Find photos of real people wearing something, then open one in the app.",
        categoryName: "Search")

    /// Sending the user into the app is the point — the snippet is a preview,
    /// not the destination.
    static var openAppWhenRun: Bool = false

    @Parameter(title: "What to look for",
               requestValueDialog: "What should the outfit have in it?")
    var query: String

    static var parameterSummary: some ParameterSummary {
        Summary("Find outfits with \(\.$query)")
    }

    @MainActor
    func perform() async throws -> some IntentResult & ProvidesDialog & ShowsSnippetView {
        let configuration = APIConfiguration()
        guard configuration.isConfigured else {
            return .result(
                dialog: "The app doesn’t have an API key yet. Open Fashion and add one in Settings.",
                view: IntentMessageView(
                    text: "No API key configured.",
                    detail: "Open Fashion → Settings to add one."))
        }

        let api = LiveFashionAPI(configuration: configuration)
        var request = OutfitSearchRequest()
        request.texts = [query]
        // Six is what fits legibly in a Siri snippet without scrolling, and
        // it is roughly as many as anyone will absorb spoken.
        request.topK = 6

        do {
            let response = try await api.outfitSearch(request)
            let results = response.results

            guard !results.isEmpty else {
                let corpus = response.corpusPosts.map { "\($0.formatted()) posts" } ?? "the corpus"
                return .result(
                    dialog: "I didn’t find any outfits with \(query). I searched \(corpus).",
                    view: IntentMessageView(
                        text: "No outfits matched “\(query)”.",
                        detail: "Searched \(corpus). That’s a real answer, not an error."))
            }

            SpokenResultsCache.shared.store(query: query, results: results)

            return .result(
                dialog: IntentDialog(stringLiteral: spoken(count: results.count)),
                view: OutfitSnippetView(results: results, query: query))
        } catch let error as FashionAPIError {
            // Outfit search is not deployed yet. Saying so beats a generic
            // "something went wrong", which would send the user looking for a
            // problem on their end.
            if case .notDeployed = error {
                return .result(
                    dialog: "Outfit search isn’t live yet. It’s still being built.",
                    view: IntentMessageView(text: "Outfit search isn’t deployed yet.",
                                            detail: "The rest of the app works."))
            }
            return .result(
                dialog: IntentDialog(stringLiteral: error.errorDescription ?? "That didn’t work."),
                view: IntentMessageView(text: error.errorDescription ?? "That didn’t work.",
                                        detail: error.recoveryAdvice))
        }
    }

    private func spoken(count: Int) -> String {
        count == 1
            ? "I found one outfit with \(query)."
            : "I found \(count) outfits with \(query). Tap one to open it."
    }
}

/// The Siri snippet.
///
/// Photographs at a legible size, two across, each a deep link. The caveat
/// about these being real people's photos with a link back is carried by the
/// app screen the link opens; the snippet keeps to what fits.
struct OutfitSnippetView: View {
    let results: [OutfitResult]
    let query: String

    private let columns = [GridItem(.adaptive(minimum: 110), spacing: 8)]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Outfits with “\(query)”")
                .font(.subheadline.weight(.medium))
                .fixedSize(horizontal: false, vertical: true)

            LazyVGrid(columns: columns, spacing: 8) {
                ForEach(results) { outfit in
                    if let url = URL(string: deepLink(for: outfit)) {
                        Link(destination: url) {
                            OutfitPhoto(url: URL(string: outfit.imageURL),
                                        accessibilityDescription: description(for: outfit))
                                .frame(height: 130)
                                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                        }
                    }
                }
            }
        }
        .padding(12)
    }

    private func deepLink(for outfit: OutfitResult) -> String {
        let id = outfit.id.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? outfit.id
        let encodedQuery = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? ""
        return "fashion://outfit/\(id)?q=\(encodedQuery)"
    }

    private func description(for outfit: OutfitResult) -> String {
        outfit.categories.isEmpty
            ? "Photograph of a person"
            : "Photograph of a person wearing " + ListFormatter.localizedString(byJoining: outfit.categories)
    }
}

struct IntentMessageView: View {
    let text: String
    var detail: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(text)
                .font(.subheadline)
                .fixedSize(horizontal: false, vertical: true)
            if let detail {
                Text(detail)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(12)
    }
}

struct FashionShortcuts: AppShortcutsProvider {
    static var appShortcuts: [AppShortcut] {
        AppShortcut(
            intent: OutfitSearchIntent(),
            phrases: [
                "Find outfits in \(.applicationName)",
                "Search outfits with \(.applicationName)",
                "Show me fits in \(.applicationName)",
            ],
            shortTitle: "Search outfits",
            systemImageName: "magnifyingglass")
    }
}

#Preview("Siri snippet") {
    OutfitSnippetView(results: StubFashionAPI.outfitFixtures(matching: ["black jacket"]),
                      query: "black jacket")
        .environment(AppEnvironment.preview)
}
