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
    .banner-site-statistics .typing-cursor {
      display: inline-block;
      width: .5em;
      animation: banner-site-statistics-cursor 1s steps(1, end) infinite;
    }
    @keyframes banner-site-statistics-cursor {
      0%, 50% { opacity: 1; }
      51%, 100% { opacity: 0; }
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
        stats.innerHTML = '<span id="banner-site-statistics-text"></span><span class="typing-cursor">_</span>';
        bannerText.appendChild(stats);

        var pvSource = document.getElementById('busuanzi_value_site_pv');
        var uvSource = document.getElementById('busuanzi_value_site_uv');
        var target = document.getElementById('banner-site-statistics-text');
        var cursor = stats.querySelector('.typing-cursor');
        var lastText = '';
        var typingTimer = null;

        function typeText(text) {
          if (!target || text === lastText) {
            return;
          }

          lastText = text;
          target.textContent = '';

          if (typingTimer) {
            window.clearInterval(typingTimer);
          }

          var index = 0;
          typingTimer = window.setInterval(function() {
            target.textContent = text.slice(0, index + 1);
            index += 1;

            if (index >= text.length) {
              window.clearInterval(typingTimer);
              typingTimer = null;
              if (cursor) {
                cursor.style.display = 'none';
              }
            } else if (cursor) {
              cursor.style.display = 'inline-block';
            }
          }, 45);
        }

        function syncAll() {
          var pv = pvSource && pvSource.textContent.trim();
          var uv = uvSource && uvSource.textContent.trim();

          if (pv && uv) {
            typeText('总访问量 ' + pv + ' 次，总访客人数 ' + uv + ' 人');
          }
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
          if (retry >= 20 || lastText) {
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
