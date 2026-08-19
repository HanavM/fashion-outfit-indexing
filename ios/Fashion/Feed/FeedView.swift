import SwiftUI

/// "Fits you could build" — outfits containing pieces like the ones you own.
///
/// Built around **one** closet item at a time rather than blending the whole
/// closet into a single feed. Two reasons, one product and one practical:
///
/// - Product: "outfits that go with my Carhartt jacket" is a question with an
///   answer. "Outfits that go with my 40 things" is not, and blending them
///   would produce a generic feed that nothing in it actually explains.
/// - Practical: each query is a request against a service that cold-starts in
///   about twenty seconds. Firing one per closet item on appear would be
///   slow and expensive for a worse result.
struct FeedView: View {
    @Environment(AppEnvironment.self) private var environment
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var selectedItemID: UUID?
    @State private var results: [OutfitResult] = []
    @State private var isLoading = false
    @State private var error: Error?
    @State private var hasLoaded = false
    @State private var clock = LoadingClock()

    private let columns = [GridItem(.adaptive(minimum: 150), spacing: Theme.Space.snug)]

    private var items: [ClosetItem] { environment.closet.items }
    private var selectedItem: ClosetItem? {
        items.first { $0.id == selectedItemID } ?? items.first
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Space.margin) {
                if environment.isDemoData { DemoDataBanner() }

                if items.isEmpty {
                    EmptyStateView(
                        title: "Add something to your closet first",
                        explanation: "This builds fits around pieces you already own. Photograph one garment and it has something to work with.",
                        actionTitle: "Go to Closet",
                        action: { environment.router.selectedTab = .closet })
                    .padding(.top, Theme.Space.section)
                } else {
                    itemSelector
                    content
                }
            }
            .padding(.bottom, Theme.Space.section)
        }
        .background(Color(.systemBackground))
        .navigationTitle("Fits")
        .task(id: selectedItem?.id) {
            guard selectedItem != nil, !hasLoaded || !isLoading else { return }
            await load()
        }
    }

    private var itemSelector: some View {
        VStack(alignment: .leading, spacing: Theme.Space.tight) {
            Text("Built around")
                .font(.subheadline.weight(.medium))
                .padding(.horizontal, Theme.Space.margin)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: Theme.Space.snug) {
                    ForEach(items) { item in
                        Button {
                            withAnimation(Theme.Motion.standard(reduceMotion: reduceMotion)) {
                                selectedItemID = item.id
                            }
                        } label: {
                            VStack(spacing: Theme.Space.tight) {
                                Group {
                                    if let image = environment.closet.image(for: item) {
                                        Image(uiImage: image)
                                            .resizable()
                                            .aspectRatio(contentMode: .fill)
                                    } else {
                                        Rectangle().fill(.quaternary)
                                    }
                                }
                                .frame(width: 64, height: 64)
                                .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.control,
                                                            style: .continuous))
                                .overlay(
                                    RoundedRectangle(cornerRadius: Theme.Radius.control,
                                                     style: .continuous)
                                        .strokeBorder(Color.accentColor,
                                                      lineWidth: selectedItem?.id == item.id ? 2 : 0))

                                Text(item.displayTitle)
                                    .font(.caption2)
                                    .lineLimit(1)
                                    .frame(width: 72)
                            }
                        }
                        .buttonStyle(.plain)
                        .minimumHitTarget()
                        .accessibilityLabel(item.displayTitle)
                        .accessibilityAddTraits(selectedItem?.id == item.id ? [.isButton, .isSelected] : .isButton)
                    }
                }
                .padding(.horizontal, Theme.Space.margin)
            }
        }
    }

    @ViewBuilder
    private var content: some View {
        if isLoading {
            LoadingView(clock: clock, label: "Finding fits")
        } else if let error {
            InlineErrorView(
                title: (error as? FashionAPIError)?.errorDescription ?? error.localizedDescription,
                advice: (error as? FashionAPIError)?.recoveryAdvice,
                retry: { Task { await load() } })
        } else if results.isEmpty && hasLoaded {
            EmptyStateView(
                title: "No fits found for that piece",
                explanation: "Nothing in the outfit corpus matched it closely enough. Try a different item — pieces from the 20 brands in the catalog tend to fare better.")
        } else {
            VStack(alignment: .leading, spacing: Theme.Space.snug) {
                Qualifier("Outfits whose garments read as similar to this piece. Similarity is visual, so it will sometimes mean “same colour and shape” rather than “same kind of thing”.")
                    .padding(.horizontal, Theme.Space.margin)

                LazyVGrid(columns: columns, spacing: Theme.Space.snug) {
                    ForEach(results) { outfit in
                        NavigationLink(value: Destination.outfit(outfit)) {
                            OutfitTile(outfit: outfit)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, Theme.Space.margin)
            }
        }
    }

    private func load() async {
        guard let item = selectedItem,
              let image = environment.closet.image(for: item),
              let encoded = ImageEncoding.base64(image) else { return }

        isLoading = true
        error = nil
        clock.start()
        defer {
            isLoading = false
            hasLoaded = true
            clock.stop()
        }

        var request = OutfitSearchRequest()
        request.images = [encoded]
        request.topK = 24
        do {
            results = try await environment.api.outfitSearch(request).results
        } catch {
            self.error = error
            results = []
        }
    }
}

#Preview("Fits") {
    NavigationStack { FeedView() }
        .environment(AppEnvironment.preview)
}

#Preview("Fits — empty closet") {
    NavigationStack { FeedView() }
        .environment(AppEnvironment(api: StubFashionAPI(),
                                    tryOn: StubTryOnService(),
                                    closet: .preview(empty: true),
                                    forceDemoData: true))
}
