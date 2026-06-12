'use strict';

hexo.extend.filter.register('theme_inject', function(injects) {
  injects.head.raw('banner-site-statistics-style', `
<% if (is_home()) { %>
  <style>
    .banner-site-statistics {
      display: inline-flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: .75rem 1rem;
      margin-top: 1rem;
      color: rgba(255, 255, 255, .92);
      font-size: .95rem;
      line-height: 1.6;
      text-shadow: 0 1px 3px rgba(0, 0, 0, .35);
    }
    .banner-site-statistics span {
      white-space: nowrap;
    }
    .banner-site-statistics strong {
      margin: 0 .25rem;
      color: #fff;
      font-weight: 600;
    }
  </style>
<% } %>
`, {}, {}, 10);

  injects.bodyEnd.raw('banner-site-statistics', `
<% if (is_home()) { %>
  <script>
    (function() {
      function initBannerSiteStatistics() {
        var bannerText = document.querySelector('#banner .banner-text');
        if (!bannerText || document.querySelector('.banner-site-statistics')) {
          return;
        }

        var stats = document.createElement('div');
        stats.className = 'banner-site-statistics';
        stats.setAttribute('aria-label', '&#20840;&#31449;&#32479;&#35745;');
        stats.innerHTML = [
          '<span>&#24635;&#35775;&#38382;&#37327;<strong id="banner-site-pv">--</strong>&#27425;</span>',
          '<span>&#24635;&#35775;&#23458;&#25968;<strong id="banner-site-uv">--</strong>&#20154;</span>'
        ].join('');
        bannerText.appendChild(stats);

        var pvSource = document.getElementById('busuanzi_value_site_pv');
        var uvSource = document.getElementById('busuanzi_value_site_uv');
        var pvTarget = document.getElementById('banner-site-pv');
        var uvTarget = document.getElementById('banner-site-uv');

        function syncValue(source, target) {
          if (source && target && source.textContent.trim()) {
            target.textContent = source.textContent.trim();
          }
        }

        function syncAll() {
          syncValue(pvSource, pvTarget);
          syncValue(uvSource, uvTarget);
        }

        syncAll();

        if (window.MutationObserver) {
          [pvSource, uvSource].forEach(function(source) {
            if (!source) {
              return;
            }
            new MutationObserver(syncAll).observe(source, {
              childList: true,
              characterData: true,
              subtree: true
            });
          });
        }

        var retry = 0;
        var timer = window.setInterval(function() {
          syncAll();
          retry += 1;
          if (retry >= 20 || (pvTarget.textContent !== '--' && uvTarget.textContent !== '--')) {
            window.clearInterval(timer);
          }
        }, 500);
      }

      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initBannerSiteStatistics);
      } else {
        initBannerSiteStatistics();
      }
    })();
  </script>
<% } %>
`, {}, {}, 10);
});
