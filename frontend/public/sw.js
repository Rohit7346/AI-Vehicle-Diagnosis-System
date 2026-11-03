/**
 * Service Worker: AI-Powered Multimodal Vehicle Fault Detection System
 * Cache Name: vehicle-diagnosis-v1
 *
 * Designed with strict protocol validation to prevent 'chrome-extension'
 * and unsupported URL scheme errors, and automatically purge stale foreign caches.
 */

const CACHE_NAME = 'vehicle-diagnosis-v1';
const STATIC_PRECACHE = [
  '/',
  '/favicon.svg',
  '/icons.svg',
];

// Helper: strictly validate if request and response can be safely cached
function isCacheable(request, response) {
  if (!request || request.method !== 'GET') {
    return false;
  }
  try {
    const url = new URL(request.url);
    // Explicitly allow ONLY http: and https: protocols
    // Excludes chrome-extension:, moz-extension:, file:, data:, blob:, etc.
    if (url.protocol !== 'http:' && url.protocol !== 'https:') {
      return false;
    }
    // Exclude API requests, Vite HMR, and WebSocket endpoints
    if (url.pathname.startsWith('/api/') || url.pathname.includes('/@vite/') || url.pathname.includes('/@react-refresh')) {
      return false;
    }
  } catch {
    return false;
  }

  // Response must be a valid 200 OK standard response
  if (!response || response.status !== 200 || response.type === 'error' || response.type === 'opaque') {
    return false;
  }
  return true;
}

// 1. Install Event: Pre-cache core shell & immediately activate
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(STATIC_PRECACHE))
      .catch((err) => console.warn('[SW] Pre-cache warning:', err))
  );
  self.skipWaiting();
});

// 2. Activate Event: Purge all stale or foreign caches (e.g. slides-editor-cache-*)
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) =>
        Promise.all(
          keys.map((key) => {
            if (key !== CACHE_NAME) {
              console.log('[SW] Purging obsolete or foreign cache storage:', key);
              return caches.delete(key);
            }
          })
        )
      )
      .then(() => self.clients.claim())
  );
});

// 3. Fetch Event: Strict protocol filtering and resilient caching
self.addEventListener('fetch', (event) => {
  const request = event.request;
  let url;

  try {
    url = new URL(request.url);
  } catch {
    return; // Ignore malformed URLs
  }

  // CRITICAL: Immediately ignore non-HTTP(S) schemes (e.g. chrome-extension://, moz-extension://)
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    return;
  }

  // Ignore non-GET methods, API routes, Vite dev server internals, and WebSockets
  if (
    request.method !== 'GET' ||
    url.pathname.startsWith('/api/') ||
    url.pathname.startsWith('/ws') ||
    url.pathname.includes('/@vite/') ||
    url.pathname.includes('/@react-refresh')
  ) {
    return;
  }

  // Navigation / HTML requests: Network-first to always serve the latest app shell
  if (request.mode === 'navigate' || request.destination === 'document') {
    event.respondWith(
      fetch(request)
        .then((response) => {
          if (isCacheable(request, response)) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((cache) => {
              cache.put(request, clone).catch(() => {});
            });
          }
          return response;
        })
        .catch(async () => {
          const cached = await caches.match(request);
          if (cached) return cached;
          return caches.match('/');
        })
    );
    return;
  }

  // Static Hashed Assets (/assets/): Cache-first with network fallback
  if (url.origin === self.location.origin && url.pathname.startsWith('/assets/')) {
    event.respondWith(
      caches.match(request).then((cached) => {
        if (cached) return cached;
        return fetch(request).then((response) => {
          if (isCacheable(request, response)) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((cache) => {
              cache.put(request, clone).catch(() => {});
            });
          }
          return response;
        });
      })
    );
    return;
  }

  // Other same-origin static requests (SVG, icons, manifests): Stale-while-revalidate
  if (url.origin === self.location.origin) {
    event.respondWith(
      caches.match(request).then((cached) => {
        const fetchPromise = fetch(request)
          .then((networkResponse) => {
            if (isCacheable(request, networkResponse)) {
              const clone = networkResponse.clone();
              caches.open(CACHE_NAME).then((cache) => {
                cache.put(request, clone).catch(() => {});
              });
            }
            return networkResponse;
          })
          .catch(() => cached);

        return cached || fetchPromise;
      })
    );
  }
});
