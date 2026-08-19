import Foundation

/// Fixture data, so every screen previews in Xcode with no network, no cold
/// start and no credentials.
///
/// The fixtures are deliberately **not** a best case. They are transcribed
/// from real responses of the live service, including the parts that are
/// unflattering, because those are the states the UI has to handle well:
///
/// - `identify` returns the *wrong* product at rank 1. This is verbatim from
///   a real call: a photo of Obey's "EST. WORKS BOLD II CREWNECK" came back
///   as a different Obey crewneck at 0.940, a pair of Uniqlo trousers at
///   0.923, and only then the correct product at 0.921. Any design that
///   shows one confident answer is wrong for this backend, and previewing
///   against this fixture makes that impossible to forget.
/// - `outfitSearch` can be told to return nothing, or two results, because
///   thin result sets are normal and honest empty states are a requirement.
struct StubFashionAPI: FashionAPI {

    enum Behaviour: Sendable {
        case normal
        /// A legitimate empty result. Not an error.
        case empty
        /// Simulates the ~20s cold start, for checking loading copy.
        case coldStart
        case failing(FashionAPIError)
        /// What the app really does today for outfit search.
        case outfitSearchNotDeployed
    }

    var behaviour: Behaviour = .normal
    /// Previews run synchronously; a real delay would just make them feel
    /// broken. Set explicitly when checking loading states.
    var artificialDelay: Duration = .zero

    init(behaviour: Behaviour = .normal, artificialDelay: Duration = .zero) {
        self.behaviour = behaviour
        self.artificialDelay = artificialDelay
    }

    private func preflight() async throws {
        if artificialDelay > .zero { try? await Task.sleep(for: artificialDelay) }
        if case .coldStart = behaviour { try? await Task.sleep(for: .seconds(20)) }
        if case .failing(let error) = behaviour { throw error }
    }

    func health() async throws -> HealthResponse {
        try await preflight()
        return HealthResponse(
            status: "ok",
            device: "cuda",
            index: .init(galleryProducts: 3922, catalogProducts: 3922, brands: StubFashionAPI.brands))
    }

    static let brands = [
        "Adidas", "Americaneagle", "Braindead", "Carhartt", "Champion", "Dickies",
        "Everlane", "Gap", "Huf", "Jcrew", "Levis", "Newbalance", "Nike",
        "Northface", "Obey", "PacSun", "Skechers", "Stussy", "Uniqlo", "Vans",
    ]

    func identify(imageData: Data, topK: Int) async throws -> IdentifyResponse {
        try await preflight()
        return StubFashionAPI.identifyFixture
    }

    /// Verbatim from the deployed service, 2026-08-18.
    static let identifyFixture = IdentifyResponse(
        results: [
            .init(rank: 1, productCode: "obey-8919243686066", brand: "Obey",
                  name: "LOWERCASE PIGMENT CREWNECK", category: "apparel item",
                  modelIdentity: "Obey apparel item", score: 0.93982),
            .init(rank: 2, productCode: "uniqlo-E462197-000-09", brand: "Uniqlo",
                  name: "Pleated Wide Pants", category: "apparel item",
                  modelIdentity: "Uniqlo apparel item", score: 0.922708),
            .init(rank: 3, productCode: "obey-8404858175666", brand: "Obey",
                  name: "EST. WORKS BOLD II CREWNECK", category: "apparel item",
                  modelIdentity: "Obey apparel item", score: 0.921429),
            .init(rank: 4, productCode: "055100862", brand: "Levis",
                  name: "510™ Skinny Men’s Jeans", category: "jeans",
                  modelIdentity: "Levi’s 510™ Skinny Men’s Jeans", score: 0.920821),
            .init(rank: 5, productCode: "everlane-M-BTM-DNM-NW-STR-BLK", brand: "Everlane",
                  name: "The Classic Straight Jean", category: "apparel item",
                  modelIdentity: "Everlane apparel item", score: 0.917069),
        ],
        confidence: 0.93982,
        rejectedOpenSet: false,
        rejectThresholdCalibrated: false,
        garmentGate: .init(score: 0.025999, threshold: 0.01,
                           looksLikeClothing: true, calibrated: true),
        predictedCategory: .init(node: "apparel item", level: "root",
                                 confidence: 1.0, bestLeaf: "sweatshirt"),
        sameModelDifferentColorwayAmbiguous: false,
        spoken: "That looks like a Obey LOWERCASE PIGMENT CREWNECK.",
        latencyMs: 1109.2)

    func query(_ request: QueryRequest) async throws -> QueryResponse {
        try await preflight()
        return QueryResponse(
            query: request.text,
            results: [
                .init(productCode: "I036804_3TU_GD", brand: "carhartt",
                      name: "Madison Ripstop Jacket", category: nil,
                      matchType: "canonical",
                      matchedLabel: "dark beech and black carhartt jacket with elasticated cuffs",
                      score: 9),
                .init(productCode: "I037133_3X7_XX", brand: "carhartt",
                      name: "Kandler Liner", category: nil, matchType: "canonical",
                      matchedLabel: "black carhartt jacket with a stand-up collar", score: 7),
                .init(productCode: "I037044_89_XX", brand: "carhartt",
                      name: "Stanwood Parka", category: nil, matchType: "canonical",
                      matchedLabel: "black carhartt jacket", score: 3),
            ],
            spoken: "I found 3 matches. The closest is a carhartt Madison Ripstop Jacket.",
            route: .init(intent: "search", equivalentEndpoint: "/search",
                         reason: "text only, so there is nothing to identify"),
            latencyMs: 15.4)
    }

    func outfitSearch(_ request: OutfitSearchRequest) async throws -> OutfitSearchResponse {
        try await preflight()
        if case .outfitSearchNotDeployed = behaviour {
            throw FashionAPIError.notDeployed("Outfit search")
        }
        if case .empty = behaviour {
            return OutfitSearchResponse(results: [],
                                        colourVocab: StubFashionAPI.colourVocab,
                                        categoryVocab: StubFashionAPI.categoryVocab,
                                        corpusPosts: 10_640)
        }

        let labels: [String] = {
            var parts = request.texts.filter { !$0.trimmed.isEmpty }
            parts.append(contentsOf: request.images.indices.map { "image \($0 + 1)" })
            return parts.isEmpty ? ["your query"] : parts
        }()

        let results = StubFashionAPI.outfitFixtures(matching: labels)
        return OutfitSearchResponse(results: Array(results.prefix(request.topK)),
                                    colourVocab: StubFashionAPI.colourVocab,
                                    categoryVocab: StubFashionAPI.categoryVocab,
                                    corpusPosts: 10_640,
                                    latencyMs: 412)
    }

    /// The stub has no photographs. Returning nil is the honest answer, and
    /// it exercises the same missing-image path the app needs anyway for a
    /// dead link or a failed download.
    func imageURL(for outfit: OutfitResult) -> URL? { nil }

    static let colourVocab = ["black", "white", "grey", "navy", "blue", "brown",
                              "beige", "green", "olive", "red", "burgundy",
                              "orange", "yellow", "purple", "pink"]

    static let categoryVocab = ["tops", "outerwear", "pants", "shorts",
                                "footwear", "headwear", "bag"]

    static func outfitFixtures(matching labels: [String]) -> [OutfitResult] {
        let seeds: [(String, String, String, String, Double, [String], [String])] = [
            ("reddit:1a2b3c", "reddit", "https://reddit.com/r/malefashionadvice/comments/1a2b3c",
             "Thrifted jacket, first fit of autumn", 0.2031,
             ["outerwear", "pants", "footwear"], ["black", "blue"]),
            ("pinterest:998877", "pinterest", "https://pinterest.com/pin/998877",
             "workwear layers", 0.1994,
             ["outerwear", "tops", "pants"], ["brown", "navy"]),
            ("reddit:4d5e6f", "reddit", "https://reddit.com/r/streetwear/comments/4d5e6f",
             "WAYWT — grey on grey", 0.1962,
             ["tops", "pants", "footwear"], ["grey"]),
            ("pinterest:112233", "pinterest", "https://pinterest.com/pin/112233",
             "denim on denim", 0.1938,
             ["outerwear", "pants"], ["blue"]),
            ("reddit:7g8h9i", "reddit", "https://reddit.com/r/malefashionadvice/comments/7g8h9i",
             "simple fit", 0.1921, ["tops", "pants", "footwear"], ["white", "black"]),
            ("pinterest:445566", "pinterest", "https://pinterest.com/pin/445566",
             "olive chore coat", 0.1904, ["outerwear", "pants"], ["olive", "beige"]),
        ]

        return seeds.enumerated().map { index, seed in
            let (id, source, postURL, title, score, categories, colors) = seed
            let parts = labels.enumerated().map { partIndex, label in
                OutfitPart(part: label,
                           kind: label.hasPrefix("image ") ? "image" : "text",
                           score: score + Double(partIndex) * 0.004,
                           matchedGarment: categories.indices.contains(partIndex)
                               ? categories[partIndex] : nil,
                           wholeFrame: !categories.indices.contains(partIndex))
            }
            return OutfitResult(id: id,
                                imageURL: "outfit_dataset/\(source)/\(id).jpg",
                                source: source,
                                postURL: postURL,
                                title: title,
                                author: "u/example\(index)",
                                score: score,
                                parts: parts,
                                categories: categories,
                                colors: colors)
        }
    }
}
