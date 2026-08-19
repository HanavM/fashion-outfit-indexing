import SwiftUI
import PhotosUI
import UIKit

/// The closet: everything you own, newest first.
///
/// A plain adaptive grid of photographs. No card chrome, no per-item borders,
/// no drop shadows — the garment photos are the content and they tile
/// directly. Titles sit under each photo in secondary type; that, plus the
/// gaps, is the entire hierarchy.
struct ClosetView: View {
    @Environment(AppEnvironment.self) private var environment
    @Environment(\.openSettings) private var openSettings
    @State private var showingCamera = false
    @State private var libraryItem: PhotosPickerItem?
    @State private var pendingImage: UIImage?

    private let columns = [GridItem(.adaptive(minimum: 108), spacing: Theme.Space.snug)]

    var body: some View {
        ScrollView {
            if environment.isDemoData { DemoDataBanner() }

            if let loadError = environment.closet.loadError {
                InlineErrorView(title: loadError,
                                advice: "Anything you add now will still be saved.")
            }

            if environment.closet.items.isEmpty {
                EmptyStateView(
                    title: "Nothing in your closet yet",
                    explanation: "Photograph a garment you own and it gets matched against \(environment.catalogDescription) Anything outside that won’t match well — you can still keep it, unidentified.",
                    actionTitle: CameraPicker.isAvailable ? "Take a photo" : nil,
                    action: CameraPicker.isAvailable ? { showingCamera = true } : nil)
                .padding(.top, Theme.Space.section)
            } else {
                LazyVGrid(columns: columns, spacing: Theme.Space.margin) {
                    ForEach(environment.closet.items) { item in
                        NavigationLink(value: Destination.closetItem(item.id)) {
                            ClosetTile(item: item)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, Theme.Space.margin)
                .padding(.top, Theme.Space.snug)
            }
        }
        .background(Color(.systemBackground))
        .navigationTitle("Closet")
        .toolbar {
            ToolbarItemGroup(placement: .topBarTrailing) {
                PhotosPicker(selection: $libraryItem, matching: .images) {
                    Label("Choose a photo", systemImage: "photo.on.rectangle")
                }
                .minimumHitTarget()

                if CameraPicker.isAvailable {
                    Button { showingCamera = true } label: {
                        Label("Take a photo", systemImage: "camera")
                    }
                    .minimumHitTarget()
                }
            }
            ToolbarItem(placement: .topBarLeading) {
                Button { openSettings() } label: {
                    Label("Settings", systemImage: "gearshape")
                }
                .minimumHitTarget()
            }
        }
        .fullScreenCover(isPresented: $showingCamera) {
            CameraPicker { pendingImage = $0 }
                .ignoresSafeArea()
        }
        .sheet(item: $pendingImage) { image in
            CaptureReviewView(image: image)
        }
        .onChange(of: libraryItem) { _, newValue in
            guard let newValue else { return }
            Task {
                pendingImage = await PhotoLibraryLoader.image(from: newValue)
                libraryItem = nil
            }
        }
    }
}

/// One garment in the grid.
private struct ClosetTile: View {
    let item: ClosetItem
    @Environment(AppEnvironment.self) private var environment

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.tight) {
            Group {
                if let image = environment.closet.image(for: item) {
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                } else {
                    ZStack {
                        Rectangle().fill(.quaternary)
                        Text("No photo")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .frame(height: 132)
            .frame(maxWidth: .infinity)
            .clipped()
            .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.photo, style: .continuous))

            Text(item.displayTitle)
                .font(.subheadline)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)

            if let brand = item.displayBrand {
                Text(brand)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            // An unendorsed guess never gets to look like a fact, even in a
            // 108pt tile.
            if item.isUnconfirmedGuess {
                Text("Unconfirmed guess")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(accessibilityLabel)
        .accessibilityHint("Opens this item")
    }

    private var accessibilityLabel: String {
        var parts = [item.displayTitle]
        if let brand = item.displayBrand { parts.append(brand) }
        if item.isUnconfirmedGuess { parts.append("unconfirmed guess") }
        if item.rejectedAllCandidates { parts.append("not identified") }
        return parts.joined(separator: ", ")
    }
}

// `UIImage` isn't `Identifiable`; this makes `.sheet(item:)` work without
// wrapping every capture in a bespoke box type.
extension UIImage: @retroactive Identifiable {
    public var id: ObjectIdentifier { ObjectIdentifier(self) }
}

#Preview("Closet — with items") {
    NavigationStack { ClosetView() }
        .environment(AppEnvironment.preview)
}

#Preview("Closet — empty") {
    NavigationStack { ClosetView() }
        .environment(AppEnvironment(api: StubFashionAPI(),
                                    tryOn: StubTryOnService(),
                                    closet: .preview(empty: true),
                                    forceDemoData: true))
}
