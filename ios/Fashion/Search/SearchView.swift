import SwiftUI
import PhotosUI
import UIKit

/// Outfit search.
///
/// The layout follows the rule that content leads: the composer is a single
/// quiet row at the top, and everything below it is photographs of people at
/// full width. Filters live behind one button rather than in a permanent bar,
/// so the chrome is one line tall no matter how much is switched on.
struct SearchView: View {
    @Environment(AppEnvironment.self) private var environment
    @Environment(\.openSettings) private var openSettings
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var model = SearchViewModel()
    @State private var showingFilters = false
    @State private var libraryItem: PhotosPickerItem?
    @FocusState private var textFieldFocused: Bool

    private let columns = [GridItem(.adaptive(minimum: 150), spacing: Theme.Space.snug)]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Space.margin) {
                if environment.isDemoData { DemoDataBanner() }
                composer
                resultsSection
            }
            .padding(.bottom, Theme.Space.section)
        }
        .background(Color(.systemBackground))
        .navigationTitle("Search")
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button { openSettings() } label: {
                    Label("Settings", systemImage: "gearshape")
                }
                .minimumHitTarget()
            }
        }
        .sheet(isPresented: $showingFilters) {
            SearchFiltersView(filters: $model.filters,
                              colourVocab: model.colourVocab.isEmpty
                                  ? StubFashionAPI.colourVocab : model.colourVocab,
                              categoryVocab: model.categoryVocab.isEmpty
                                  ? StubFashionAPI.categoryVocab : model.categoryVocab)
        }
        .onChange(of: libraryItem) { _, newValue in
            guard let newValue else { return }
            Task {
                if let image = await PhotoLibraryLoader.image(from: newValue) {
                    withAnimation(Theme.Motion.standard(reduceMotion: reduceMotion)) {
                        model.parts.append(.image(image))
                    }
                }
                libraryItem = nil
            }
        }
        .onAppear {
            if SearchComposition.shared.hasStaged {
                model.parts.append(contentsOf: SearchComposition.shared.consume())
                textFieldFocused = true
            }
        }
        .onChange(of: environment.router.pendingSearchText) { _, newValue in
            guard let newValue, !newValue.isEmpty else { return }
            model.reset()
            model.loadSpokenQuery(newValue, using: environment.api)
            environment.router.pendingSearchText = nil
        }
    }

    // MARK: Composer

    private var composer: some View {
        VStack(alignment: .leading, spacing: Theme.Space.snug) {
            if !model.parts.isEmpty {
                partsList
            }

            HStack(spacing: Theme.Space.snug) {
                PhotosPicker(selection: $libraryItem, matching: .images) {
                    Image(systemName: "photo.badge.plus")
                        .imageScale(.large)
                }
                .minimumHitTarget()
                .accessibilityLabel("Add a photo to the query")

                TextField("worn with baggy jeans", text: $model.draftText)
                    .textFieldStyle(.plain)
                    .focused($textFieldFocused)
                    .submitLabel(.search)
                    .onSubmit { runSearch() }
                    .frame(minHeight: Theme.minimumTouchTarget)

                Button("Search") { runSearch() }
                    .buttonStyle(.borderedProminent)
                    .minimumHitTarget()
                    .disabled(!model.canSearch)
            }
            .padding(.horizontal, Theme.Space.margin)

            Button {
                showingFilters = true
            } label: {
                HStack(spacing: Theme.Space.tight) {
                    Image(systemName: model.filters.isActive
                        ? "line.3.horizontal.decrease.circle.fill"
                        : "line.3.horizontal.decrease.circle")
                    Text(model.filters.summary)
                        .lineLimit(1)
                }
                .font(.subheadline)
            }
            .minimumHitTarget()
            .padding(.horizontal, Theme.Space.margin)
            .accessibilityLabel("Filters. \(model.filters.isActive ? model.filters.summary + " active" : "none active")")

            if model.parts.count > 1 {
                Qualifier("Each part is matched to a different garment in the photo.")
                    .padding(.horizontal, Theme.Space.margin)
            }
        }
        .padding(.top, Theme.Space.snug)
    }

    private var partsList: some View {
        VStack(alignment: .leading, spacing: Theme.Space.tight) {
            ForEach(model.parts) { part in
                HStack(spacing: Theme.Space.snug) {
                    if case .image(_, let image, _) = part {
                        Image(uiImage: image)
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                            .frame(width: 40, height: 40)
                            .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.control / 2,
                                                        style: .continuous))
                            .accessibilityHidden(true)
                    } else {
                        Image(systemName: "text.quote")
                            .frame(width: 40)
                            .foregroundStyle(.secondary)
                            .accessibilityHidden(true)
                    }

                    Text(part.displayText)
                        .font(.subheadline)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .fixedSize(horizontal: false, vertical: true)

                    Button {
                        withAnimation(Theme.Motion.standard(reduceMotion: reduceMotion)) {
                            model.remove(part)
                        }
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.tertiary)
                    }
                    .minimumHitTarget()
                    .accessibilityLabel("Remove \(part.displayText) from the query")
                }
                .frame(minHeight: Theme.minimumTouchTarget)
            }
        }
        .padding(.horizontal, Theme.Space.margin)
    }

    // MARK: Results

    @ViewBuilder
    private var resultsSection: some View {
        if model.isLoading {
            LoadingView(clock: model.clock, label: "Searching outfits")
        } else if let error = model.error {
            InlineErrorView(
                title: (error as? FashionAPIError)?.errorDescription ?? error.localizedDescription,
                advice: (error as? FashionAPIError)?.recoveryAdvice,
                retry: { runSearch() })
        } else if !model.results.isEmpty {
            LazyVGrid(columns: columns, spacing: Theme.Space.snug) {
                ForEach(model.results) { outfit in
                    NavigationLink(value: Destination.outfit(outfit)) {
                        OutfitTile(outfit: outfit)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, Theme.Space.margin)
        } else if model.hasSearched {
            EmptyStateView(
                title: "No outfits matched",
                explanation: emptyExplanation)
        } else {
            EmptyStateView(
                title: "Search real outfits",
                explanation: "Add a photo of a garment, some words for what it should be worn with, or both. Results are photographs of real people, each linking back to where it came from.")
        }
    }

    private var emptyExplanation: String {
        let corpus = model.corpusPosts.map { "\($0.formatted()) posts" } ?? "the whole corpus"
        return "Nothing in \(corpus) matched all the parts of that query. That’s a real answer, not a failure — try removing a part or loosening the filters."
    }

    private func runSearch() {
        textFieldFocused = false
        model.search(using: environment.api)
    }
}

/// One outfit in the results grid.
///
/// The photograph is the whole tile. The score sits underneath in secondary
/// type rather than as an overlay badge, because an overlay would cover the
/// picture and imply a precision the score doesn't have — the spread between
/// rank 1 and rank 50 on this encoder is about 0.015.
struct OutfitTile: View {
    let outfit: OutfitResult
    @Environment(AppEnvironment.self) private var environment

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.tight) {
            OutfitPhoto(url: environment.api.imageURL(for: outfit),
                        accessibilityDescription: photoDescription)
                .frame(height: 210)
                .frame(maxWidth: .infinity)
                .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.photo, style: .continuous))

            HStack(alignment: .firstTextBaseline) {
                Text(outfit.source?.capitalized ?? "Outfit")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                ScoreLabel(outfit.score)
                    .font(.caption)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(photoDescription)
        .accessibilityHint("Opens this outfit")
    }

    private var photoDescription: String {
        var description = "Photograph of a person"
        if !outfit.categories.isEmpty {
            description += " wearing " + ListFormatter.localizedString(byJoining: outfit.categories)
            description += " — garments detected automatically, unverified"
        }
        if let source = outfit.source { description += ". From \(source.capitalized)" }
        return description
    }
}

#Preview("Search — start") {
    NavigationStack { SearchView() }
        .environment(AppEnvironment.preview)
}

#Preview("Search — no results") {
    NavigationStack { SearchView() }
        .environment(AppEnvironment.preview(behaviour: .empty))
}

#Preview("Search — outfit API not deployed") {
    NavigationStack { SearchView() }
        .environment(AppEnvironment.preview(behaviour: .outfitSearchNotDeployed))
}
