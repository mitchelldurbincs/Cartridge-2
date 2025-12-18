import { mount, unmount } from 'svelte';
import App from './App.svelte';
import LossOverTimePage from './LossOverTimePage.svelte';

type Component = typeof App | typeof LossOverTimePage;

let currentApp: ReturnType<typeof mount> | null = null;

function getRoute(): string {
  const hash = window.location.hash.slice(1) || '/';
  return hash;
}

function getComponent(route: string): Component {
  switch (route) {
    case '/loss-over-time':
      return LossOverTimePage;
    default:
      return App;
  }
}

function renderRoute() {
  const target = document.getElementById('app')!;
  const route = getRoute();
  const Component = getComponent(route);

  if (currentApp) {
    unmount(currentApp);
  }

  currentApp = mount(Component, { target });
}

// Initial render
renderRoute();

// Listen for hash changes
window.addEventListener('hashchange', renderRoute);

export default currentApp;
