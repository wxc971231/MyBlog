'use strict';

hexo.extend.filter.register('theme_inject', function(injects) {
  injects.footer.raw('page-view-statistics', `
<% if (!is_post()) { %>
  <div class="statistics page-statistics">
    <span id="busuanzi_container_page_pv" style="display: none">
      本页访问量
      <span id="busuanzi_value_page_pv"></span>
      次
    </span>
  </div>
<% } %>
`, {}, {}, 5);
});
