import Foundation

/// The bridge between a Siri answer and a tap.
///
/// An App Intent runs, gets outfit results, and speaks. The user then taps
/// one. Because there is no endpoint that returns a single outfit by id, the
/// intent parks what it found here and the deep link reads it back. Kept
/// small and disposable on purpose: this is a handoff buffer, not a database.
final class SpokenResultsCache: @unchecked Sendable {
    static let shared = SpokenResultsCache()

    private let lock = NSLock()
    private let fileURL: URL

    private init() {
        let base = FileManager.default.urls(for: .applicationSupportDirectory,
                                            in: .userDomainMask)[0]
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        fileURL = base.appendingPathComponent("spoken-results.json")
    }

    private struct Entry: Codable {
        var query: String
        var storedAt: Date
        var results: [OutfitResult]
    }

    func store(query: String, results: [OutfitResult]) {
        lock.lock()
        defer { lock.unlock() }
        let entry = Entry(query: query, storedAt: .now, results: results)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        guard let data = try? encoder.encode(entry) else { return }
        try? data.write(to: fileURL, options: .atomic)
    }

    func outfit(id: String) -> OutfitResult? {
        lock.lock()
        defer { lock.unlock() }
        guard let data = try? Data(contentsOf: fileURL) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let entry = try? decoder.decode(Entry.self, from: data) else { return nil }
        return entry.results.first { $0.id == id }
    }
}
