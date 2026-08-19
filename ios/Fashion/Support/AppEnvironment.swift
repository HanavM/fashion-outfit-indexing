import SwiftUI

/// The single composition point: which API implementation is live, which
/// try-on service, and the closet.
///
/// The important behaviour here is the **honest fallback**. If no API key can
/// be found, the app does not crash, does not show a login wall, and does not
/// silently show a broken screen. It runs against `StubFashionAPI` and sets
/// `isDemoData`, which every screen surfaces as a banner. Fake data is
/// allowed; unlabelled fake data is not.
@MainActor
@Observable
final class AppEnvironment {
    let configuration: APIConfiguration
    private(set) var api: FashionAPI
    let tryOn: TryOnService
    let closet: ClosetStore
    let router = Router()

    /// True when the data on screen came from fixtures rather than the
    /// service. Drives the banner.
    private(set) var isDemoData: Bool

    /// Filled in by a background `/health` call. Used to tell the user the
    /// real size and shape of the catalog in empty states, rather than
    /// leaving them to guess why their Zara jacket came back as Uniqlo.
    private(set) var health: HealthResponse?

    init(configuration: APIConfiguration = APIConfiguration(),
         api: FashionAPI? = nil,
         tryOn: TryOnService? = nil,
         closet: ClosetStore? = nil,
         forceDemoData: Bool? = nil) {
        self.configuration = configuration
        if let api {
            self.api = api
            self.isDemoData = forceDemoData ?? (api is StubFashionAPI)
        } else if configuration.isConfigured {
            self.api = LiveFashionAPI(configuration: configuration)
            self.isDemoData = forceDemoData ?? false
        } else {
            self.api = StubFashionAPI()
            self.isDemoData = forceDemoData ?? true
        }
        self.tryOn = tryOn ?? AzureTryOnService()
        self.closet = closet ?? ClosetStore()
    }

    /// Called after the user enters a key in Settings, so the app switches to
    /// the live service without a relaunch.
    func reloadAPI() {
        if configuration.isConfigured {
            api = LiveFashionAPI(configuration: configuration)
            isDemoData = false
        } else {
            api = StubFashionAPI()
            isDemoData = true
        }
    }

    func refreshHealth() async {
        health = try? await api.health()
    }

    /// The catalog's boundary, in one sentence, from live data when we have
    /// it and from the known deployment when we don't.
    var catalogDescription: String {
        let count = health?.productCount ?? 3_922
        let brands = health?.brands.count ?? 20
        return "\(count.formatted()) products from \(brands) US brands, men’s clothing only."
    }

    @MainActor
    static var preview: AppEnvironment {
        AppEnvironment(api: StubFashionAPI(),
                       tryOn: StubTryOnService(),
                       closet: .preview(),
                       forceDemoData: true)
    }

    /// `closet` is optional rather than defaulted to `.preview()` because a
    /// default argument is evaluated in a nonisolated context, and the
    /// preview store is main-actor isolated.
    @MainActor
    static func preview(behaviour: StubFashionAPI.Behaviour,
                        closet: ClosetStore? = nil) -> AppEnvironment {
        AppEnvironment(api: StubFashionAPI(behaviour: behaviour),
                       tryOn: StubTryOnService(),
                       closet: closet ?? .preview(),
                       forceDemoData: true)
    }
}
