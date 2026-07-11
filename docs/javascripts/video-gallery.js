(function () {
  var youtubeAllow =
    "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share";

  function loadVideo(button) {
    var videoId = button.dataset.youtubeId;
    if (!videoId) {
      return;
    }

    var iframe = document.createElement("iframe");
    iframe.src = "https://www.youtube.com/embed/" + videoId + "?autoplay=1";
    iframe.title = button.dataset.title || "YouTube video";
    iframe.allow = youtubeAllow;
    iframe.referrerPolicy = "strict-origin-when-cross-origin";
    iframe.allowFullscreen = true;
    iframe.loading = "lazy";
    iframe.style.aspectRatio = button.dataset.aspect || "16 / 9";

    var wrapper = button.closest(".video-embed");
    if (wrapper) {
      wrapper.classList.add("video-embed-loaded");
      wrapper.replaceChildren(iframe);
    }
  }

  function initVideoGallery(root) {
    var buttons = root.querySelectorAll(".video-lite:not([data-video-ready])");

    buttons.forEach(function (button) {
      button.dataset.videoReady = "true";
      button.addEventListener("click", function () {
        loadVideo(button);
      });
    });
  }

  if (window.document$) {
    window.document$.subscribe(function () {
      initVideoGallery(document);
    });
  } else {
    document.addEventListener("DOMContentLoaded", function () {
      initVideoGallery(document);
    });
  }
})();
