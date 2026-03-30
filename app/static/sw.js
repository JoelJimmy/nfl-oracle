// sw.js — NFL Oracle Service Worker
// Handles push notifications for game alerts and weekly best bets

const CACHE_NAME = "nfl-oracle-v1";

// ── Push notification handler ─────────────────────────────────────────────────
self.addEventListener("push", (event) => {
  if (!event.data) return;

  let data = {};
  try {
    data = event.data.json();
  } catch (e) {
    data = { title: "NFL Oracle", body: event.data.text() };
  }

  const options = {
    body:    data.body    || "New prediction available",
    icon:    data.icon    || "/static/icon-192.png",
    badge:   data.badge   || "/static/icon-96.png",
    tag:     data.tag     || "nfl-oracle",
    data:    { url: data.url || "/" },
    actions: [
      { action: "view", title: "View Prediction" },
      { action: "dismiss", title: "Dismiss" },
    ],
    requireInteraction: false,
    silent: false,
  };

  event.waitUntil(
    self.registration.showNotification(data.title || "NFL Oracle 🏈", options)
  );
});

// ── Notification click ────────────────────────────────────────────────────────
self.addEventListener("notificationclick", (event) => {
  event.notification.close();

  if (event.action === "dismiss") return;

  const url = event.notification.data?.url || "/";
  event.waitUntil(
    clients.matchAll({ type: "window", includeUncontrolled: true }).then((clientList) => {
      for (const client of clientList) {
        if (client.url === url && "focus" in client) {
          return client.focus();
        }
      }
      if (clients.openWindow) {
        return clients.openWindow(url);
      }
    })
  );
});

// ── Basic offline cache ───────────────────────────────────────────────────────
self.addEventListener("install", (event) => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(clients.claim());
});