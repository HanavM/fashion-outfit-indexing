import Foundation
import Security

/// Where the API key comes from, and in what order.
///
/// The key is never in source and never in git. It is resolved at runtime
/// from, in priority order:
///
/// 1. the `FASHION_API_KEY` **environment variable** — set it on the Xcode
///    scheme, which stores it in the user's own `.xcuserdata`, not in the
///    shared project file. Best for development.
/// 2. the **Keychain**, written by the in-app Settings screen. Best for
///    running on a device without touching the project at all.
/// 3. `FashionAPIKey` in Info.plist, substituted at build time from the
///    gitignored `Config/Secrets.xcconfig`.
///
/// If none of the three produce a key, the app does not crash and does not
/// pretend: `AppEnvironment` falls back to the stub and shows a banner
/// saying the data is demo data.
@Observable
final class APIConfiguration {
    private static let keychainAccount = "FASHION_API_KEY"
    private static let keychainService = "com.hanavm.fashion"

    private(set) var apiKey: String?

    /// The **identifier** service: `/health`, `/identify`, `/query`. It knows
    /// the catalog.
    let baseURL: URL

    /// The **outfit search** service: `/outfit_search`, `/photo`. It knows the
    /// corpus of real outfit photos, and it is a different Modal app on a
    /// different host — outfit search is CPU-only, the identifier needs an
    /// A10G, so they were never going to be one deployment.
    ///
    /// These were collapsed into a single `baseURL` until 2026-09-11, which
    /// meant every `/outfit_search` call went to the identifier host and came
    /// back 404. The client dutifully translated that into "Outfit search is
    /// not deployed" and the app said so, convincingly, for a month — while
    /// the service was up the whole time.
    let outfitBaseURL: URL

    /// Where the currently-active key came from. Surfaced in Settings so it
    /// is obvious which of the three sources actually won — otherwise a
    /// stale Keychain entry silently beating a fresh xcconfig is a very
    /// confusing half hour.
    enum Source: String {
        case environment = "Scheme environment variable"
        case keychain = "Keychain"
        case buildSettings = "Secrets.xcconfig"
        case none = "Not configured"
    }

    private(set) var source: Source

    init(bundle: Bundle = .main,
         environment: [String: String] = ProcessInfo.processInfo.environment) {
        baseURL = APIConfiguration.resolveURL(
            environment: environment["FASHION_API_BASE_URL"],
            plistValue: bundle.object(forInfoDictionaryKey: "FashionAPIBaseURL") as? String,
            fallback: APIConfiguration.fallbackBaseURL)

        outfitBaseURL = APIConfiguration.resolveURL(
            environment: environment["FASHION_OUTFIT_BASE_URL"],
            plistValue: bundle.object(forInfoDictionaryKey: "FashionOutfitBaseURL") as? String,
            fallback: APIConfiguration.fallbackOutfitBaseURL)

        if let fromEnvironment = environment["FASHION_API_KEY"]?.trimmed, !fromEnvironment.isEmpty {
            apiKey = fromEnvironment
            source = .environment
        } else if let fromKeychain = APIConfiguration.readKeychain(), !fromKeychain.isEmpty {
            apiKey = fromKeychain
            source = .keychain
        } else if let fromPlist = (bundle.object(forInfoDictionaryKey: "FashionAPIKey") as? String)?.trimmed,
                  !fromPlist.isEmpty {
            apiKey = fromPlist
            source = .buildSettings
        } else {
            apiKey = nil
            source = .none
        }
    }

    /// Environment first, then Info.plist, then the literal — the same
    /// precedence the key uses, so a scheme variable can repoint either
    /// service at a local `serve` without touching the project.
    ///
    /// The plist value is checked for emptiness because the xcconfig declares
    /// empty defaults: without them the `$(...)` reference survives into the
    /// built plist as literal text, and `URL(string:)` would happily accept
    /// `$(FASHION_OUTFIT_BASE_URL)` as a relative URL.
    private static func resolveURL(environment: String?,
                                   plistValue: String?,
                                   fallback: String) -> URL {
        for candidate in [environment?.trimmed, plistValue?.trimmed] {
            guard let candidate, !candidate.isEmpty,
                  let url = URL(string: candidate), url.scheme != nil else { continue }
            return url
        }
        return URL(string: fallback)!
    }

    /// The base URLs are duplicated here as literals only so that a clone with
    /// no xcconfig at all still points somewhere sensible. They are public
    /// hostnames, not secrets.
    private static let fallbackBaseURL =
        "https://hanavm--fashion-serve-fashionservice-api.modal.run"

    private static let fallbackOutfitBaseURL =
        "https://hanavm--outfit-serve-outfitservice-api.modal.run"

    var isConfigured: Bool { apiKey?.isEmpty == false }

    // MARK: Keychain

    func store(apiKey newKey: String) {
        let trimmed = newKey.trimmed
        guard !trimmed.isEmpty else { return }
        APIConfiguration.writeKeychain(trimmed)
        apiKey = trimmed
        source = .keychain
    }

    func clearStoredKey() {
        APIConfiguration.deleteKeychain()
        apiKey = nil
        source = .none
    }

    private static func baseQuery() -> [String: Any] {
        [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: keychainService,
         kSecAttrAccount as String: keychainAccount]
    }

    private static func readKeychain() -> String? {
        var query = baseQuery()
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne

        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    private static func writeKeychain(_ value: String) {
        deleteKeychain()
        var query = baseQuery()
        query[kSecValueData as String] = Data(value.utf8)
        query[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        SecItemAdd(query as CFDictionary, nil)
    }

    private static func deleteKeychain() {
        SecItemDelete(baseQuery() as CFDictionary)
    }
}
