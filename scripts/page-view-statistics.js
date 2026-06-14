'use strict';

hexo.extend.filter.register('theme_inject', function(injects) {
  injects.head.raw('page-view-statistics-style', `
<% if (!is_post() && !is_home()) { %>
  <style>
    .banner-page-statistics {
      display: inline-flex;
      justify-content: center;
      margin-top: .85rem;
      color: rgba(255, 255, 255, .92);
      font-size: .95rem;
      line-height: 1.6;
      text-shadow: 0 1px 3px rgba(0, 0, 0, .35);
    }
    .banner-page-statistics strong {
      margin: 0 .25rem;
      color: #fff;
      font-weight: 600;
    }
  </style>
<% } %>
`, {}, {}, 10);

  injects.footer.raw('page-view-statistics', `
<% if (!is_post()) { %>
  <div class="statistics page-statistics">
    <span id="busuanzi_container_page_pv" style="display: none">
      &#26412;&#39029;&#35775;&#38382;&#37327;
      <span id="busuanzi_value_page_pv"></span>
      &#27425;
    </span>
  </div>
<% } %>
`, {}, {}, 5);

  injects.bodyEnd.raw('banner-page-view-statistics', `
<% if (!is_post() && !is_home()) { %>
  <script>
    (function() {
      function initBannerPageStatistics() {
        var bannerText = document.querySelector('#banner .banner-text');
        if (!bannerText || document.querySelector('.banner-page-statistics')) {
          return;
        }

        var stats = document.createElement('div');
        stats.className = 'banner-page-statistics';
        stats.innerHTML = '&#26412;&#39029;&#35775;&#38382;&#37327;<strong id="banner-page-pv">--</strong>&#27425;';
        bannerText.appendChild(stats);

        var source = document.getElementById('busuanzi_value_page_pv');
        var target = document.getElementById('banner-page-pv');

        function syncValue() {
          if (source && target && source.textContent.trim()) {
            target.textContent = source.textContent.trim();
          }
        }

        syncValue();

        if (window.MutationObserver && source) {
          new MutationObserver(syncValue).observe(source, {
            childList: true,
            characterData: true,
            subtree: true
          });
        }

        var retry = 0;
        var timer = window.setInterval(function() {
          syncValue();
          retry += 1;
          if (retry >= 20 || target.textContent !== '--') {
            window.clearInterval(timer);
          }
        }, 500);
      }

      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initBannerPageStatistics);
      } else {
        initBannerPageStatistics();
      }
    })();
  </script>
<% } %>
`, {}, {}, 10);
});
