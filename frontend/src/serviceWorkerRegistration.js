/**
 * Service Worker Registration & Foreign Cache Purge Module
 * Ensures seamless PWA support while proactively evicting any stale
 * or conflicting service workers and caches from other local projects.
 */

export function register() {
  if (typeof window === 'undefined' || !('serviceWorker' in navigator)) {
    return;
  }

  // 1. Proactively purge any foreign or obsolete caches (e.g. previous slide/presentation caches)
  if ('caches' in window) {
    caches.keys().then((keys) => {
      keys.forEach((key) => {
        if (key.includes('slides') || key.includes('presentation') || (key.startsWith('vehicle-diagnosis-') && key !== 'vehicle-diagnosis-v1')) {
          console.log('[App] Evicting foreign/stale cache storage:', key);
          caches.delete(key);
        }
      });
    }).catch(() => {});
  }

  // 2. Register current project service worker
  window.addEventListener('load', () => {
    const swUrl = '/sw.js';

    navigator.serviceWorker
      .getRegistrations()
      .then((registrations) => {
        for (const reg of registrations) {
          // If a foreign service worker was registered under this origin, unregister it
          const scriptUrl = reg.active?.scriptURL || reg.installing?.scriptURL || reg.waiting?.scriptURL || '';
          if (scriptUrl && (scriptUrl.includes('slides') || scriptUrl.includes('ppt') || !scriptUrl.endsWith('/sw.js'))) {
            console.log('[App] Unregistering foreign service worker:', scriptUrl);
            reg.unregister();
          }
        }

        // Register the vehicle diagnosis service worker
        navigator.serviceWorker
          .register(swUrl)
          .then((registration) => {
            console.log('[App] Service Worker active with scope:', registration.scope);

            registration.onupdatefound = () => {
              const installingWorker = registration.installing;
              if (!installingWorker) return;

              installingWorker.onstatechange = () => {
                if (installingWorker.state === 'installed') {
                  if (navigator.serviceWorker.controller) {
                    console.log('[App] New content available; please refresh.');
                  } else {
                    console.log('[App] Content cached for offline use.');
                  }
                }
              };
            };
          })
          .catch((error) => {
            console.warn('[App] Service Worker registration failed:', error);
          });
      })
      .catch((error) => {
        console.warn('[App] Error inspecting existing service workers:', error);
      });
  });
}

export function unregister() {
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.ready
      .then((registration) => {
        registration.unregister();
      })
      .catch((error) => {
        console.warn('[App] Error unregistering service worker:', error);
      });
  }
}
