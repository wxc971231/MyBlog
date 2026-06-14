'use strict';

hexo.extend.filter.register('theme_inject', function(injects) {
  injects.head.raw('page-view-statistics-style', `
<% if (!is_post() && !is_home()) { %>
  <style>
    .banner-page-statistics {
      display: inline-flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: .4rem 1rem;
      margin-top: .85rem;
      color: rgba(255, 255, 255, .92);
      font-size: .95rem;
      line-height: 1.6;
      text-shadow: 0 1px 3px rgba(0, 0, 0, .35);
    }
    .banner-page-statistics .typing-cursor {
      display: inline-block;
      width: .5em;
      animation: banner-page-statistics-cursor 1s steps(1, end) infinite;
    }
    @keyframes banner-page-statistics-cursor {
      0%, 50% { opacity: 1; }
      51%, 100% { opacity: 0; }
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
        stats.innerHTML = '<span id="banner-page-statistics-text"></span><span class="typing-cursor">_</span>';
        bannerText.appendChild(stats);

        var pvSource = document.getElementById('busuanzi_value_page_pv');
        var uvSource = document.getElementById('busuanzi_value_site_uv');
        var target = document.getElementById('banner-page-statistics-text');
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

        function syncValue() {
          var pv = pvSource && pvSource.textContent.trim();
          var uv = uvSource && uvSource.textContent.trim();

          if (pv && uv) {
            typeText('本页访问量 ' + pv + ' 次，总访客人数 ' + uv + ' 人');
          }
        }

        syncValue();

        if (window.MutationObserver) {
          [pvSource, uvSource].forEach(function(source) {
            if (!source) {
              return;
            }
            new MutationObserver(syncValue).observe(source, {
              childList: true,
              characterData: true,
              subtree: true
            });
          });
        }

        var retry = 0;
        var timer = window.setInterval(function() {
          syncValue();
          retry += 1;
          if (retry >= 20 || lastText) {
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
