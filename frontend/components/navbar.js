/**
 * Navigation bar and mobile menu controller with live backend status indicator.
 */

export function initNavbar(onNavigate) {
  const navLinks = document.querySelectorAll("[data-nav-target]");
  const mobileMenuBtn = document.getElementById("mobile-menu-btn");
  const mobileMenu = document.getElementById("mobile-menu");

  navLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const target = link.getAttribute("data-nav-target");
      if (target) {
        onNavigate(target);
        if (mobileMenu && !mobileMenu.classList.contains("hidden")) {
          mobileMenu.classList.add("hidden");
        }
      }
    });
  });

  if (mobileMenuBtn && mobileMenu) {
    mobileMenuBtn.addEventListener("click", () => {
      mobileMenu.classList.toggle("hidden");
    });
  }
}

export function setActiveNav(pageId) {
  const desktopLinks = document.querySelectorAll(".nav-link-desktop");
  const mobileLinks = document.querySelectorAll(".nav-link-mobile");

  desktopLinks.forEach((el) => {
    const active = el.getAttribute("data-nav-target") === pageId;
    el.classList.toggle("bg-emerald-50", active);
    el.classList.toggle("text-emerald-700", active);
    el.classList.toggle("font-semibold", active);
    el.classList.toggle("text-slate-600", !active);
  });

  mobileLinks.forEach((el) => {
    const active = el.getAttribute("data-nav-target") === pageId;
    el.classList.toggle("bg-emerald-50", active);
    el.classList.toggle("text-emerald-700", active);
    el.classList.toggle("font-semibold", active);
    el.classList.toggle("text-slate-600", !active);
  });
}

export function updateHealthPill(healthData, error = null) {
  const pill = document.getElementById("nav-health-pill");
  const dot = document.getElementById("nav-health-dot");
  const text = document.getElementById("nav-health-text");
  if (!pill || !dot || !text) return;

  if (error || !healthData) {
    pill.className =
      "inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium bg-rose-50 text-rose-700 border border-rose-200";
    dot.className = "w-2 h-2 rounded-full bg-rose-500";
    text.textContent = "Backend Offline";
    return;
  }

  if (healthData.model_loaded && healthData.database_connected) {
    pill.className =
      "inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium bg-emerald-50 text-emerald-800 border border-emerald-200";
    dot.className = "w-2 h-2 rounded-full bg-emerald-500 animate-pulse";
    text.textContent = `AI Ready · ${healthData.num_classes} Classes`;
  } else {
    pill.className =
      "inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium bg-amber-50 text-amber-800 border border-amber-200";
    dot.className = "w-2 h-2 rounded-full bg-amber-500";
    text.textContent = "Model Not Trained";
  }
}
