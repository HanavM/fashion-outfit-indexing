import SwiftUI
import UIKit

@MainActor
@Observable
final class SearchViewModel {
    var parts: [QueryPart] = []
    var draftText: String = ""
    var filters = SearchFilters()

    private(set) var results: [OutfitResult] = []
    private(set) var colourVocab: [String] = []
    private(set) var categoryVocab: [String] = []
    private(set) var corpusPosts: Int?
    private(set) var isLoading = false
    private(set) var error: Error?
    /// Distinguishes "you haven't searched yet" from "that search found
    /// nothing". They need different words on screen.
    private(set) var hasSearched = false

    let clock = LoadingClock()
    private var task: Task<Void, Never>?

    var canSearch: Bool {
        !parts.isEmpty || !draftText.trimmed.isEmpty
    }

    func addDraftText() {
        let value = draftText.trimmed
        guard !value.isEmpty else { return }
        parts.append(.text(value))
        draftText = ""
    }

    func remove(_ part: QueryPart) {
        parts.removeAll { $0.id == part.id }
    }

    func search(using api: FashionAPI) {
        addDraftText()
        guard canSearch else { return }

        task?.cancel()
        let request = buildRequest()
        task = Task { [weak self] in
            guard let self else { return }
            self.isLoading = true
            self.error = nil
            self.clock.start()
            defer {
                self.isLoading = false
                self.clock.stop()
            }
            do {
                let response = try await api.outfitSearch(request)
                guard !Task.isCancelled else { return }
                self.results = response.results
                // The vocabularies come from the response, so the colour and
                // category controls can only ever offer values the index
                // actually understands.
                if !response.colourVocab.isEmpty { self.colourVocab = response.colourVocab }
                if !response.categoryVocab.isEmpty { self.categoryVocab = response.categoryVocab }
                self.corpusPosts = response.corpusPosts
                self.hasSearched = true
            } catch {
                guard !Task.isCancelled else { return }
                self.error = error
                self.results = []
                self.hasSearched = true
            }
        }
    }

    private func buildRequest() -> OutfitSearchRequest {
        var request = OutfitSearchRequest()
        for part in parts {
            switch part {
            case .image(_, let image, _):
                if let encoded = ImageEncoding.base64(image) { request.images.append(encoded) }
            case .text(_, let value):
                request.texts.append(value)
            }
        }
        request.topK = 24
        request.colourName = filters.colourRGB == nil ? filters.colourName : nil
        request.colourRGB = filters.colourRGB
        request.dropNonUS = filters.usOnly ? true : nil
        request.dropWomens = filters.mensOnly ? true : nil
        request.skinTone = filters.skinReferenceImage == nil ? filters.skinTone : nil
        if let reference = filters.skinReferenceImage {
            request.skinImageBase64 = ImageEncoding.base64(reference)
        }
        // The category filter rides as an extra phrase rather than a field:
        // the agreed contract has no category parameter, and adding one
        // client-side would be inventing API. A phrase is honest and the
        // ranker already handles it.
        if let category = filters.category {
            request.texts.append(category)
        }
        return request
    }

    /// Used by the Siri intent and the deep-link fallback.
    func loadSpokenQuery(_ text: String, using api: FashionAPI) {
        parts = [.text(text)]
        search(using: api)
    }

    func reset() {
        task?.cancel()
        parts = []
        draftText = ""
        results = []
        error = nil
        hasSearched = false
    }
}
