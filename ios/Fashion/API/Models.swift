import Foundation

// MARK: - Health

/// `GET /health`. Used for one thing in the UI: telling the user how big the
/// searchable world actually is, in the empty state, before they waste a
/// query on something outside it.
struct HealthResponse: Codable, Sendable, Equatable {
    struct Index: Codable, Sendable, Equatable {
        var galleryProducts: Int?
        var catalogProducts: Int?
        var brands: [String]?

        enum CodingKeys: String, CodingKey {
            case galleryProducts = "gallery_products"
            case catalogProducts = "catalog_products"
            case brands
        }
    }

    var status: String
    var device: String?
    var index: Index?

    var brands: [String] { index?.brands ?? [] }
    var productCount: Int { index?.galleryProducts ?? index?.catalogProducts ?? 0 }
}

// MARK: - Identify

/// One candidate product from `/identify`.
///
/// `rank` matters as much as `score` here: the API returns a ranked list and
/// the UI shows several of them, because on this backend the top-ranked
/// answer is regularly not the right one. Verified against the live service:
/// a photo of an Obey "EST. WORKS BOLD II CREWNECK" returned that exact
/// product at rank 3 (0.921), behind a different Obey crewneck (0.940) and a
/// pair of Uniqlo trousers (0.923).
struct IdentifiedProduct: Codable, Sendable, Identifiable, Equatable {
    var rank: Int?
    var productCode: String
    var brand: String?
    var name: String?
    var category: String?
    var modelIdentity: String?
    var score: Double

    var id: String { productCode }

    /// What to actually put on screen. The catalog's `name` is missing or
    /// junk for a minority of records, so fall back rather than render blank.
    var displayName: String {
        if let name, !name.isEmpty { return name }
        if let modelIdentity, !modelIdentity.isEmpty { return modelIdentity }
        return productCode
    }

    enum CodingKeys: String, CodingKey {
        case rank
        case productCode = "product_code"
        case brand, name, category, score
        case modelIdentity = "model_identity"
    }
}

/// The gate that decides whether the photo even shows clothing.
struct GarmentGate: Codable, Sendable, Equatable {
    var score: Double?
    var threshold: Double?
    var looksLikeClothing: Bool?
    var calibrated: Bool?

    enum CodingKeys: String, CodingKey {
        case score, threshold, calibrated
        case looksLikeClothing = "looks_like_clothing"
    }
}

struct PredictedCategory: Codable, Sendable, Equatable {
    var node: String?
    var level: String?
    var confidence: Double?
    var bestLeaf: String?

    enum CodingKeys: String, CodingKey {
        case node, level, confidence
        case bestLeaf = "best_leaf"
    }
}

struct IdentifyResponse: Codable, Sendable, Equatable {
    var results: [IdentifiedProduct]
    var confidence: Double?
    var rejectedOpenSet: Bool?
    /// The backend reports this as `false`: open-set rejection is
    /// uncalibrated, so "this is not in the catalog" never fires on its own.
    /// The UI has to say so instead of implying the absence of a rejection
    /// means the match is real.
    var rejectThresholdCalibrated: Bool?
    var garmentGate: GarmentGate?
    var predictedCategory: PredictedCategory?
    var sameModelDifferentColorwayAmbiguous: Bool?
    var spoken: String?
    var latencyMs: Double?

    var top: IdentifiedProduct? { results.first }
    var alternatives: [IdentifiedProduct] { Array(results.dropFirst()) }

    enum CodingKeys: String, CodingKey {
        case results, confidence, spoken
        case rejectedOpenSet = "rejected_open_set"
        case rejectThresholdCalibrated = "reject_threshold_calibrated"
        case garmentGate = "garment_gate"
        case predictedCategory = "predicted_category"
        case sameModelDifferentColorwayAmbiguous = "same_model_different_colorway_ambiguous"
        case latencyMs = "latency_ms"
    }
}

// MARK: - Query (catalog)

struct QueryRequest: Codable, Sendable {
    var imageBase64: String?
    var text: String?
    var topK: Int?

    enum CodingKeys: String, CodingKey {
        case imageBase64 = "image_base64"
        case text
        case topK = "top_k"
    }
}

struct QueryRoute: Codable, Sendable, Equatable {
    var intent: String?
    var equivalentEndpoint: String?
    var reason: String?

    enum CodingKeys: String, CodingKey {
        case intent, reason
        case equivalentEndpoint = "equivalent_endpoint"
    }
}

struct QueryResult: Codable, Sendable, Identifiable, Equatable {
    var productCode: String
    var brand: String?
    var name: String?
    var category: String?
    var matchType: String?
    var matchedLabel: String?
    var score: Double

    var id: String { productCode }

    var displayName: String {
        if let name, !name.isEmpty { return name }
        return productCode
    }

    /// Verified against the live API: the text-only route returns an integer
    /// keyword score (9, 7, 3 — not 0...1), while image routes return a
    /// cosine similarity. Rendering both the same way would misrepresent one
    /// of them, so the scale travels with the value.
    var scoreScale: ScoreLabel.Scale {
        matchType == "canonical" || matchType == "lexical" ? .keywordMatch : .similarity
    }

    enum CodingKeys: String, CodingKey {
        case productCode = "product_code"
        case brand, name, category, score
        case matchType = "match_type"
        case matchedLabel = "matched_label"
    }
}

struct QueryResponse: Codable, Sendable, Equatable {
    var query: String?
    var results: [QueryResult]
    var spoken: String?
    var route: QueryRoute?
    var garmentGate: GarmentGate?
    var latencyMs: Double?

    enum CodingKeys: String, CodingKey {
        case query, results, spoken, route
        case garmentGate = "garment_gate"
        case latencyMs = "latency_ms"
    }
}

// MARK: - Outfit search

/// `POST /outfit_search`.
///
/// This endpoint is **not deployed yet** — it is being built in parallel.
/// Everything here is coded to the agreed contract and reached only through
/// `FashionAPI`, so swapping in the real thing is a change to
/// `LiveFashionAPI` and nothing else.
struct OutfitSearchRequest: Codable, Sendable {
    /// Any number of photos and any number of phrases, in any mix. Each part
    /// is matched to a *different* garment in the result photo, which is why
    /// this is two arrays rather than one image and one string.
    var images: [String] = []
    var texts: [String] = []
    var topK: Int = 24

    var colourName: String?
    var colourRGB: [Int]?
    var skinImageBase64: String?
    /// 0...1 position on a *relative* control. See `SkinToneSlider` for why
    /// this is never labelled with tone categories or numbers.
    var skinTone: Double?
    var dropNonUS: Bool?
    var dropWomens: Bool?

    var isEmpty: Bool { images.isEmpty && texts.allSatisfy { $0.trimmed.isEmpty } }

    enum CodingKeys: String, CodingKey {
        case images, texts
        case topK = "top_k"
        case colourName = "colour_name"
        case colourRGB = "colour_rgb"
        case skinImageBase64 = "skin_image_base64"
        case skinTone = "skin_tone"
        case dropNonUS = "drop_non_us"
        case dropWomens = "drop_womens"
    }
}

/// How one part of the query (a photo, or a phrase) was satisfied by one
/// garment in the result photo.
struct OutfitPart: Codable, Sendable, Identifiable, Equatable {
    /// The part as the user expressed it: a phrase, or "image 1".
    var part: String
    var kind: String?
    var score: Double
    /// The garment this part claimed. Nil when the part matched the whole
    /// frame rather than a detected garment.
    var matchedGarment: String?
    var wholeFrame: Bool?

    var id: String { part }

    var matchedDescription: String {
        if let matchedGarment, !matchedGarment.isEmpty { return matchedGarment }
        return "the whole photo"
    }

    enum CodingKeys: String, CodingKey {
        case part, kind, score
        case matchedGarment = "matched_garment"
        case wholeFrame = "whole_frame"
    }
}

/// One outfit photo.
///
/// Decoding is deliberately lenient. The deployed contract promises `id` and
/// `image_url`; the local engine this was developed against emits `rel`,
/// `path` and `post_id` instead. Rather than break on whichever ships, the
/// initialiser accepts either and derives what is missing.
struct OutfitResult: Codable, Sendable, Identifiable, Equatable {
    var id: String
    var imageURL: String
    var source: String?
    var postURL: String?
    var title: String?
    var author: String?
    var score: Double
    var parts: [OutfitPart] = []
    /// Garment categories detected in the photo. Unvalidated model output —
    /// wherever these are displayed, that qualifier is displayed too.
    var categories: [String] = []
    var colors: [String] = []

    enum CodingKeys: String, CodingKey {
        case id, source, score, parts, title, author, categories, colors
        case imageURL = "image_url"
        case postURL = "post_url"
        case rel, path
        case postID = "post_id"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let rel = try container.decodeIfPresent(String.self, forKey: .rel)
        let path = try container.decodeIfPresent(String.self, forKey: .path)
        let postID = try container.decodeIfPresent(String.self, forKey: .postID)
        let url = try container.decodeIfPresent(String.self, forKey: .imageURL)

        imageURL = url ?? rel ?? path ?? ""
        id = try container.decodeIfPresent(String.self, forKey: .id)
            ?? rel ?? postID ?? imageURL
        source = try container.decodeIfPresent(String.self, forKey: .source)
        postURL = try container.decodeIfPresent(String.self, forKey: .postURL)
        title = try container.decodeIfPresent(String.self, forKey: .title)
        author = try container.decodeIfPresent(String.self, forKey: .author)
        score = try container.decodeIfPresent(Double.self, forKey: .score) ?? 0
        parts = try container.decodeIfPresent([OutfitPart].self, forKey: .parts) ?? []
        categories = try container.decodeIfPresent([String].self, forKey: .categories) ?? []
        colors = try container.decodeIfPresent([String].self, forKey: .colors) ?? []
    }

    init(id: String, imageURL: String, source: String? = nil, postURL: String? = nil,
         title: String? = nil, author: String? = nil, score: Double,
         parts: [OutfitPart] = [], categories: [String] = [], colors: [String] = []) {
        self.id = id
        self.imageURL = imageURL
        self.source = source
        self.postURL = postURL
        self.title = title
        self.author = author
        self.score = score
        self.parts = parts
        self.categories = categories
        self.colors = colors
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(imageURL, forKey: .imageURL)
        try container.encodeIfPresent(source, forKey: .source)
        try container.encodeIfPresent(postURL, forKey: .postURL)
        try container.encodeIfPresent(title, forKey: .title)
        try container.encodeIfPresent(author, forKey: .author)
        try container.encode(score, forKey: .score)
        try container.encode(parts, forKey: .parts)
        try container.encode(categories, forKey: .categories)
        try container.encode(colors, forKey: .colors)
    }
}

struct OutfitSearchResponse: Codable, Sendable, Equatable {
    var results: [OutfitResult]
    /// Colour names the backend actually understands. The colour filter is
    /// populated from this rather than from a hardcoded list, so the control
    /// can never offer a colour the index cannot honour.
    var colourVocab: [String] = []
    var categoryVocab: [String] = []
    /// How many posts were searched. Shown in the empty state, so "no
    /// results" reads as a fact about a corpus of known size.
    var corpusPosts: Int?
    var latencyMs: Double?

    enum CodingKeys: String, CodingKey {
        case results
        case colourVocab = "colour_vocab"
        case categoryVocab = "category_vocab"
        case corpusPosts = "corpus_posts"
        case latencyMs = "latency_ms"
    }

    init(results: [OutfitResult], colourVocab: [String] = [],
         categoryVocab: [String] = [], corpusPosts: Int? = nil,
         latencyMs: Double? = nil) {
        self.results = results
        self.colourVocab = colourVocab
        self.categoryVocab = categoryVocab
        self.corpusPosts = corpusPosts
        self.latencyMs = latencyMs
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        results = try container.decodeIfPresent([OutfitResult].self, forKey: .results) ?? []
        colourVocab = try container.decodeIfPresent([String].self, forKey: .colourVocab) ?? []
        categoryVocab = try container.decodeIfPresent([String].self, forKey: .categoryVocab) ?? []
        corpusPosts = try container.decodeIfPresent(Int.self, forKey: .corpusPosts)
        latencyMs = try container.decodeIfPresent(Double.self, forKey: .latencyMs)
    }
}

extension String {
    var trimmed: String { trimmingCharacters(in: .whitespacesAndNewlines) }
}
