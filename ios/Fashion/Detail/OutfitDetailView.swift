import SwiftUI

/// One outfit, large.
///
/// The photograph gets the top of the screen at full width with nothing on
/// top of it — no scrim, no floating glass panel, no overlaid title. Every
/// piece of metadata sits below it in the scroll. That is the "content leads"
/// rule taken literally, and it is also the respectful way to show a
/// photograph of a person who did not opt into being here.
struct OutfitDetailView: View {
    let outfit: OutfitResult

    @Environment(AppEnvironment.self) private var environment
    @State private var showingTryOn = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Space.margin) {
                OutfitPhoto(url: environment.api.imageURL(for: outfit),
                            accessibilityDescription: photoDescription,
                            contentMode: .fit)
                    .frame(maxWidth: .infinity)
                    .frame(minHeight: 320)

                VStack(alignment: .leading, spacing: Theme.Space.margin) {
                    if let title = outfit.title, !title.isEmpty {
                        Text(title)
                            .font(.title3.weight(.semibold))
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    // Attribution is not tucked away at the bottom. It is
                    // directly under the photograph, above everything the app
                    // has to say about it.
                    SourceLink(outfit: outfit)

                    if !outfit.parts.isEmpty {
                        matchedParts
                    }

                    DetectedGarments(categories: outfit.categories, colors: outfit.colors)

                    ShoppableGarments(outfit: outfit)

                    Divider()

                    tryOnEntry
                }
                .padding(.horizontal, Theme.Space.margin)
            }
            .padding(.bottom, Theme.Space.section)
        }
        .background(Color(.systemBackground))
        .navigationTitle("Outfit")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showingTryOn) {
            TryOnView(outfit: outfit)
        }
    }

    /// Why this photo came back — which part of the query claimed which
    /// garment. This is the one place the ranking explains itself, and it
    /// matters because a multi-part query that silently satisfies both parts
    /// with the same jacket is the failure mode users need to be able to spot.
    private var matchedParts: some View {
        VStack(alignment: .leading, spacing: Theme.Space.tight) {
            Text("Why this matched")
                .font(.subheadline.weight(.medium))
            ForEach(outfit.parts) { part in
                HStack(alignment: .firstTextBaseline, spacing: Theme.Space.tight) {
                    Text("“\(part.part)”")
                        .fixedSize(horizontal: false, vertical: true)
                    Text("→ \(part.matchedDescription)")
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer()
                    ScoreLabel(part.score)
                }
                .font(.footnote)
                .accessibilityElement(children: .combine)
                .accessibilityLabel("\(part.part) matched \(part.matchedDescription), similarity \(String(format: "%.2f", part.score))")
            }
            Qualifier("Scores here sit in a narrow band on this encoder — the gap between the best and the fiftieth result is usually a couple of hundredths. Treat the ordering as a suggestion.")
        }
    }

    @ViewBuilder
    private var tryOnEntry: some View {
        VStack(alignment: .leading, spacing: Theme.Space.tight) {
            Button {
                showingTryOn = true
            } label: {
                Label("See it on you", systemImage: "wand.and.stars")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .minimumHitTarget()

            Qualifier("Makes a generated picture — not a photograph of you, and not a photograph of this outfit.")
        }
    }

    private var photoDescription: String {
        var description = "Photograph of a person"
        if !outfit.categories.isEmpty {
            description += " wearing " + ListFormatter.localizedString(byJoining: outfit.categories)
            description += ", detected automatically and unverified"
        }
        if let author = outfit.author { description += ". Posted by \(author)" }
        return description
    }
}

/// Product links for the pieces in this outfit.
///
/// An important honesty constraint shapes this: the app cannot say "this
/// exact jacket is a Carhartt Detroit". The outfit corpus is unlabelled, and
/// the detected garment is a category, not a product. So the section asks the
/// catalog for products *of that kind* and says exactly that — "similar
/// pieces in the catalog", never "the item in this photo".
private struct ShoppableGarments: View {
    let outfit: OutfitResult
    @Environment(AppEnvironment.self) private var environment

    @State private var results: [String: [QueryResult]] = [:]
    @State private var loading: Set<String> = []

    /// Categories the user doesn't already own something in. Rough, and
    /// labelled as rough — closet categories come from the same imperfect
    /// classifier.
    private var missing: [String] {
        let owned = Set(environment.closet.items.compactMap { $0.category?.lowercased() })
        return outfit.categories.filter { !owned.contains($0.lowercased()) }
    }

    var body: some View {
        if !missing.isEmpty {
            VStack(alignment: .leading, spacing: Theme.Space.snug) {
                Text("Similar pieces you could buy")
                    .font(.subheadline.weight(.medium))
                Qualifier("These are catalog products of the same kind of garment. They are not the item in the photograph — nothing here knows what that actually is.")

                ForEach(missing, id: \.self) { category in
                    VStack(alignment: .leading, spacing: Theme.Space.tight) {
                        HStack {
                            Text(category.capitalized)
                                .font(.subheadline)
                            Spacer()
                            if loading.contains(category) {
                                ProgressView()
                            } else if results[category] == nil {
                                Button("Find") { Task { await load(category) } }
                                    .font(.subheadline)
                                    .minimumHitTarget()
                            }
                        }

                        ForEach(results[category] ?? []) { product in
                            HStack(alignment: .firstTextBaseline, spacing: Theme.Space.snug) {
                                VStack(alignment: .leading, spacing: Theme.Space.hairline) {
                                    Text(product.displayName)
                                        .font(.footnote)
                                        .fixedSize(horizontal: false, vertical: true)
                                    if let brand = product.brand {
                                        Text(brand.capitalized)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
                                ScoreLabel(product.score, scale: product.scoreScale)
                            }
                            .padding(.vertical, Theme.Space.hairline)
                            .accessibilityElement(children: .combine)
                        }

                        if let found = results[category], found.isEmpty {
                            Text("Nothing in the catalog for that.")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                    }
                    Divider()
                }
            }
        }
    }

    private func load(_ category: String) async {
        loading.insert(category)
        defer { loading.remove(category) }
        let colour = outfit.colors.first.map { "\($0) " } ?? ""
        let response = try? await environment.api.query(
            QueryRequest(imageBase64: nil, text: "\(colour)\(category)", topK: 3))
        results[category] = response?.results ?? []
    }
}

#Preview("Outfit detail") {
    NavigationStack {
        OutfitDetailView(outfit: StubFashionAPI.outfitFixtures(matching: ["black jacket", "baggy jeans"])[0])
    }
    .environment(AppEnvironment.preview)
}
