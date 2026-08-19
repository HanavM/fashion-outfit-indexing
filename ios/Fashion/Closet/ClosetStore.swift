import Foundation
import UIKit

/// The closet, on disk.
///
/// **Why Codable + FileManager rather than SwiftData.** Three reasons, all
/// specific to this app rather than general preference:
///
/// 1. The closet is a single-user list of tens of items with no relationships
///    and no queries beyond "all of them, newest first". SwiftData's value is
///    querying, relationships and change tracking across contexts; none of
///    that is being bought here.
/// 2. The heavy part of an item is its photograph, which does not belong in a
///    database row either way. Images are written as individual JPEGs and the
///    record stores a filename. Once the images are files, the metadata being
///    one small JSON file next to them is simpler than running a store
///    alongside them.
/// 3. The identification schema is still moving — the outfit API isn't even
///    deployed. A JSON document with optional fields absorbs that; a
///    SwiftData model asks for a migration each time.
///
/// The trade this accepts: no partial loads, and a full rewrite of the
/// manifest on every mutation. At closet scale (tens to low hundreds of
/// items, a few KB of JSON) that is genuinely free. If the closet ever grows
/// into thousands of items or gains sync, this is the thing to revisit.
@Observable
@MainActor
final class ClosetStore {
    private(set) var items: [ClosetItem] = []
    private(set) var loadError: String?

    private let directory: URL
    private var imagesDirectory: URL { directory.appendingPathComponent("Images", isDirectory: true) }
    private var manifestURL: URL { directory.appendingPathComponent("closet.json") }

    /// In-memory image cache, so a scrolling grid doesn't re-decode JPEGs.
    private var imageCache: [String: UIImage] = [:]

    init(directory: URL? = nil) {
        if let directory {
            self.directory = directory
        } else {
            let base = FileManager.default.urls(for: .applicationSupportDirectory,
                                                in: .userDomainMask)[0]
            self.directory = base.appendingPathComponent("Closet", isDirectory: true)
        }
        createDirectories()
        load()
    }

    private func createDirectories() {
        try? FileManager.default.createDirectory(at: imagesDirectory,
                                                 withIntermediateDirectories: true)
    }

    // MARK: Reading

    private func load() {
        guard FileManager.default.fileExists(atPath: manifestURL.path) else { return }
        do {
            let data = try Data(contentsOf: manifestURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            items = try decoder.decode([ClosetItem].self, from: data)
                .sorted { $0.capturedAt > $1.capturedAt }
        } catch {
            // Surfaced in the UI rather than swallowed: silently showing an
            // empty closet to someone who has fifty items in it is the worst
            // possible failure here.
            loadError = "Your closet couldn’t be read from disk."
        }
    }

    func image(for item: ClosetItem) -> UIImage? {
        if let cached = imageCache[item.imageFilename] { return cached }
        let url = imagesDirectory.appendingPathComponent(item.imageFilename)
        guard let image = UIImage(contentsOfFile: url.path) else { return nil }
        imageCache[item.imageFilename] = image
        return image
    }

    // MARK: Writing

    @discardableResult
    func add(image: UIImage, candidates: [IdentifiedProduct]) -> ClosetItem? {
        guard let data = ImageEncoding.prepared(image) else { return nil }
        let filename = "\(UUID().uuidString).jpg"
        do {
            try data.write(to: imagesDirectory.appendingPathComponent(filename))
        } catch {
            loadError = "That photo couldn’t be saved."
            return nil
        }
        let item = ClosetItem(imageFilename: filename, candidates: candidates)
        imageCache[filename] = image
        items.insert(item, at: 0)
        save()
        return item
    }

    func update(_ item: ClosetItem) {
        guard let index = items.firstIndex(where: { $0.id == item.id }) else { return }
        items[index] = item
        save()
    }

    func confirm(_ item: ClosetItem, productCode: String) {
        var updated = item
        updated.confirmedProductCode = productCode
        updated.rejectedAllCandidates = false
        update(updated)
    }

    func rejectAllCandidates(for item: ClosetItem) {
        var updated = item
        updated.confirmedProductCode = nil
        updated.rejectedAllCandidates = true
        update(updated)
    }

    func rename(_ item: ClosetItem, to nickname: String) {
        var updated = item
        updated.nickname = nickname.trimmed.isEmpty ? nil : nickname.trimmed
        update(updated)
    }

    func delete(_ item: ClosetItem) {
        items.removeAll { $0.id == item.id }
        imageCache[item.imageFilename] = nil
        try? FileManager.default.removeItem(
            at: imagesDirectory.appendingPathComponent(item.imageFilename))
        save()
    }

    private func save() {
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(items)
            // Atomic: a crash mid-write must not leave a truncated manifest
            // that loses the whole closet.
            try data.write(to: manifestURL, options: .atomic)
            loadError = nil
        } catch {
            loadError = "Your closet couldn’t be saved."
        }
    }
}

// MARK: - Previews

extension ClosetStore {
    /// A store backed by a throwaway directory, pre-populated with items that
    /// cover the three states worth designing for: confirmed, unconfirmed
    /// guess, and "none of these were right".
    static func preview(empty: Bool = false) -> ClosetStore {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ClosetPreview-\(UUID().uuidString)", isDirectory: true)
        let store = ClosetStore(directory: directory)
        guard !empty else { return store }

        let fixtures = StubFashionAPI.identifyFixture.results
        store.items = [
            ClosetItem(imageFilename: "preview-0.jpg",
                       capturedAt: .now.addingTimeInterval(-3600),
                       candidates: fixtures,
                       confirmedProductCode: "obey-8404858175666"),
            ClosetItem(imageFilename: "preview-1.jpg",
                       capturedAt: .now.addingTimeInterval(-86_400),
                       candidates: fixtures),
            ClosetItem(imageFilename: "preview-2.jpg",
                       capturedAt: .now.addingTimeInterval(-172_800),
                       candidates: fixtures,
                       rejectedAllCandidates: true,
                       nickname: "Grandad’s wool overshirt"),
        ]
        return store
    }
}
