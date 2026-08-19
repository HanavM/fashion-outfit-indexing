import Foundation

/// One garment the user owns.
///
/// The identification is stored as *the whole candidate list*, not as a
/// single resolved product. That is a deliberate consequence of the
/// backend's measured behaviour: the top-ranked candidate is often not the
/// right one, so the item keeps every candidate it was offered and records
/// which one the user picked, plus whether they picked it at all. A closet
/// item that was never confirmed says so, forever, rather than hardening a
/// guess into a fact.
struct ClosetItem: Codable, Identifiable, Equatable, Sendable {
    let id: UUID
    /// Filename inside the closet's image directory. The image itself is not
    /// in this struct — see `ClosetStore` for why.
    var imageFilename: String
    var capturedAt: Date

    /// Everything `/identify` offered, in rank order.
    var candidates: [IdentifiedProduct]
    /// Which candidate the user actually endorsed. Nil means they haven't.
    var confirmedProductCode: String?
    /// Set when the user says none of the candidates is right. The item stays
    /// in the closet — you still own the garment — it just has no catalog
    /// identity, which is the honest state for anything outside 20 US men's
    /// brands.
    var rejectedAllCandidates: Bool

    /// What the user calls it. The only field that is unambiguously theirs.
    var nickname: String?

    var confirmed: IdentifiedProduct? {
        guard let confirmedProductCode else { return nil }
        return candidates.first { $0.productCode == confirmedProductCode }
    }

    /// The best available label, degrading honestly.
    var displayTitle: String {
        if let nickname, !nickname.trimmed.isEmpty { return nickname }
        if let confirmed { return confirmed.displayName }
        if rejectedAllCandidates { return "Unidentified item" }
        if let first = candidates.first { return first.displayName }
        return "Item"
    }

    var displayBrand: String? {
        if rejectedAllCandidates { return nil }
        return confirmed?.brand ?? candidates.first?.brand
    }

    var category: String? {
        confirmed?.category ?? candidates.first?.category
    }

    /// True when a guess is on screen that nobody has vouched for.
    var isUnconfirmedGuess: Bool {
        confirmedProductCode == nil && !rejectedAllCandidates && !candidates.isEmpty
    }

    init(id: UUID = UUID(),
         imageFilename: String,
         capturedAt: Date = .now,
         candidates: [IdentifiedProduct] = [],
         confirmedProductCode: String? = nil,
         rejectedAllCandidates: Bool = false,
         nickname: String? = nil) {
        self.id = id
        self.imageFilename = imageFilename
        self.capturedAt = capturedAt
        self.candidates = candidates
        self.confirmedProductCode = confirmedProductCode
        self.rejectedAllCandidates = rejectedAllCandidates
        self.nickname = nickname
    }
}
