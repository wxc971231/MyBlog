'use strict';

hexo.extend.filter.register('theme_inject', function(injects) {
  const config = hexo.theme.config.code_collapse || {};
  if (config.enable === false) {
    return;
  }

  const maxLines = Math.max(parseInt(config.max_lines, 10) || 20, 5);

  injects.head.raw('code-collapse-style', `
  <style>
    .code-collapse {
      position: relative;
      margin: 1rem 0;
      border-radius: 4px;
    }
    .code-collapse-content {
      position: relative;
      transition: max-height .28s ease;
    }
    .code-collapse.is-collapsed .code-collapse-content {
      max-height: var(--code-collapse-height);
      overflow: hidden;
    }
    .code-collapse.is-collapsed .code-collapse-content::after {
      content: '';
      position: absolute;
      right: 0;
      bottom: 0;
      left: 0;
      height: 3rem;
      pointer-events: none;
      background: linear-gradient(to bottom, transparent, var(--highlight-bg-color));
    }
    .code-collapse > .code-collapse-content > figure.highlight,
    .code-collapse > .code-collapse-content > .code-wrapper,
    .code-collapse > .code-collapse-content > pre {
      margin: 0;
    }
    .code-collapse-toggle {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: .35rem;
      width: 100%;
      min-height: 2.25rem;
      padding: .35rem .75rem;
      border: 0;
      border-top: 1px solid rgba(127, 127, 127, .22);
      border-radius: 0 0 4px 4px;
      color: var(--sec-text-color);
      background: var(--highlight-bg-color);
      font-size: .875rem;
      cursor: pointer;
      transition: color .2s ease, background-color .2s ease;
    }
    .code-collapse-toggle:hover,
    .code-collapse-toggle:focus-visible {
      color: var(--link-hover-color);
      background-color: rgba(127, 127, 127, .12);
      outline: none;
    }
    .code-collapse-toggle-icon {
      display: inline-block;
      transition: transform .28s ease;
    }
    .code-collapse.is-expanded .code-collapse-toggle-icon {
      transform: rotate(180deg);
    }
    @media (prefers-reduced-motion: reduce) {
      .code-collapse-content,
      .code-collapse-toggle-icon {
        transition: none;
      }
    }
  </style>
`, {}, {}, 20);

  injects.bodyEnd.raw('code-collapse-script', `
  <script>
    (function() {
      var maxLines = ${maxLines};

      function getRoot(pre) {
        return pre.closest('figure.highlight') || pre.closest('.code-wrapper') || pre;
      }

      function getLineCount(root, pre) {
        var highlightedLines = root.querySelectorAll('td.gutter span.line');
        if (highlightedLines.length) {
          return highlightedLines.length;
        }

        var code = pre.querySelector('code');
        var text = code ? code.textContent : pre.textContent;
        return text.replace(/\\n$/, '').split('\\n').length;
      }

      function getCollapsedHeight(pre) {
        var style = window.getComputedStyle(pre);
        var lineHeight = parseFloat(style.lineHeight);
        var fontSize = parseFloat(style.fontSize) || 16;
        var paddingTop = parseFloat(style.paddingTop) || 0;
        var paddingBottom = parseFloat(style.paddingBottom) || 0;

        if (!Number.isFinite(lineHeight)) {
          lineHeight = fontSize * 1.5;
        }

        return Math.ceil(lineHeight * maxLines + paddingTop + paddingBottom);
      }

      function enhanceCodeBlocks() {
        var roots = new Set();

        document.querySelectorAll('.markdown-body pre').forEach(function(pre) {
          if (pre.querySelector('code.mermaid') || pre.closest('.code-collapse')) {
            return;
          }

          var root = getRoot(pre);
          if (roots.has(root) || getLineCount(root, pre) <= maxLines) {
            return;
          }
          roots.add(root);

          var wrapper = document.createElement('div');
          var content = document.createElement('div');
          var button = document.createElement('button');
          var collapsedHeight = getCollapsedHeight(pre);

          wrapper.className = 'code-collapse is-collapsed';
          wrapper.style.setProperty('--code-collapse-height', collapsedHeight + 'px');
          content.className = 'code-collapse-content';
          button.className = 'code-collapse-toggle';
          button.type = 'button';
          button.setAttribute('aria-expanded', 'false');
          button.innerHTML = '<span class="code-collapse-toggle-icon" aria-hidden="true">&#9660;</span><span>展开全部</span>';

          root.parentNode.insertBefore(wrapper, root);
          content.appendChild(root);
          wrapper.appendChild(content);
          wrapper.appendChild(button);

          button.addEventListener('click', function() {
            var expanded = wrapper.classList.toggle('is-expanded');
            wrapper.classList.toggle('is-collapsed', !expanded);
            button.setAttribute('aria-expanded', String(expanded));
            button.lastElementChild.textContent = expanded ? '收起代码' : '展开全部';

            if (!expanded && wrapper.getBoundingClientRect().top < 0) {
              wrapper.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
          });
        });
      }

      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', enhanceCodeBlocks);
      } else {
        enhanceCodeBlocks();
      }
    })();
  </script>
`, {}, {}, 20);
});
