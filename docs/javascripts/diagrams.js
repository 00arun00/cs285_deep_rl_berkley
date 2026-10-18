/* Shared Mermaid presentation D. Keep the public entry point and internal Dagre
 * adapter pinned together: upgrading Mermaid requires a rendered-diagram review.
 * Only flowcharts receive D geometry; other diagram types keep native geometry.
 * No diagram source is rewritten. The original fence remains available below.
 */
(() => {
  const base = 'https://cdn.jsdelivr.net/npm/mermaid@12.2.0/dist/';
  let library, serial = 0, running = false, pending = false;
  const diagrams = new Map();
  const nodeCSS = `.node rect{rx:12px;ry:12px;stroke-width:1px}.node text,.nodeLabel{font-size:15px!important;font-weight:600!important}.node tspan[font-weight="normal"]{font-weight:600!important}.node.fieldnotes-decision rect{fill:var(--md-default-bg-color)!important;stroke:var(--md-typeset-a-color)!important;stroke-dasharray:3 3}.edgeLabel text{font-size:13px!important;font-weight:500!important}.edgeLabel rect,.edgeLabel .label rect{rx:12px;ry:12px;opacity:1!important;fill:var(--md-default-bg-color)!important;stroke:var(--md-typeset-a-color)!important;stroke-width:1px}.flowchart-link{stroke-width:1px!important;stroke-linecap:round;stroke-linejoin:round}`;

  async function load() {
    if (!library) library = (async () => {
      const { default: mermaid } = await import(base + 'mermaid.esm.min.mjs');
      mermaid.registerLayoutLoaders([{ name: 'fieldnotes-dagre', loader: async () => {
        // Mermaid's layout API is public; this exact engine chunk path is internal.
        const engine = await import(base + 'chunks/mermaid.esm.min/dagre-JKPHJXCU.mjs');
        if (typeof engine.render !== 'function') throw new Error('Incompatible Mermaid layout engine');
        return { ...engine, render: async (graph, ...args) => {
          if (!Array.isArray(graph.nodes)) throw new Error('Incompatible Mermaid graph metadata');
          for (const node of graph.nodes) {
            if (node.isGroup) continue;
            if (['diamond', 'diam', 'decision', 'question'].includes(node.shape)) {
              node.shape = 'rect';
              node.cssClasses = `${node.cssClasses || ''} fieldnotes-decision`;
            }
            if (['rect', 'squareRect', 'rounded', 'roundedRect'].includes(node.shape)) {
              node.padding = 20;
              node.height = Math.max(56, node.height || 0);
            }
          }
          return engine.render(graph, ...args);
        }};
      }}]);
      return mermaid;
    })().catch(error => { library = undefined; throw error; });
    return library;
  }

  function configuration(flow) {
    const c = getComputedStyle(document.body), v = key => c.getPropertyValue(key).trim();
    const bg = v('--md-default-bg-color'), fg = v('--md-default-fg-color');
    const border = v('--field-border'), line = v('--md-default-fg-color--light'), node = v('--field-nav-active');
    return {
      startOnLoad: false, securityLevel: 'strict', suppressErrorRendering: true,
      theme: 'base', look: 'classic', layout: flow ? 'fieldnotes-dagre' : 'dagre',
      themeVariables: {
        darkMode: document.body.dataset.mdColorScheme === 'slate', background: bg,
        primaryColor: node, primaryTextColor: fg, primaryBorderColor: border,
        secondaryColor: v('--md-code-bg-color'), secondaryTextColor: fg, secondaryBorderColor: border,
        tertiaryColor: bg, tertiaryTextColor: fg, tertiaryBorderColor: border,
        lineColor: line, textColor: fg, edgeLabelBackground: bg, mainBkg: node,
        fontFamily: 'Avenir Next, Segoe UI, sans-serif', fontSize: '15px',
        actorBkg: node, actorBorder: border, actorTextColor: fg, actorLineColor: line,
        signalColor: line, signalTextColor: fg, labelBoxBkgColor: bg, labelBoxBorderColor: border,
        labelTextColor: fg, loopTextColor: fg, noteBkgColor: v('--md-code-bg-color'),
        noteTextColor: fg, noteBorderColor: border, activationBkgColor: node,
        activationBorderColor: border, classText: fg, attributeBackgroundColor: bg,
        stateLabelColor: fg, compositeBackground: bg, compositeTitleBackground: node, transitionColor: line
      },
      sequence: { useMaxWidth: false, actorFontSize: 15, messageFontSize: 14, noteFontSize: 14, wrap: true },
      class: { useMaxWidth: false, hideEmptyMembersBox: true }, state: { useMaxWidth: false },
      flowchart: { htmlLabels: false, useMaxWidth: false, nodeSpacing: 50, rankSpacing: 50, curve: 'basis', wrappingWidth: 200 },
      ...(flow ? { htmlLabels: false, themeCSS: nodeCSS } : {})
    };
  }

  function button(text, action) {
    const b = document.createElement('button'); b.type = 'button'; b.textContent = text;
    b.addEventListener('click', action); return b;
  }

  function mount(target, svg, title, pills) {
    // HTML parsing preserves foreignObject HTML (including br); XML parsing does not.
    const template = document.createElement('template'); template.innerHTML = svg;
    const root = template.content.querySelector('svg');
    if (!root || template.content.querySelector('parsererror')) throw new Error('Invalid diagram output');
    root.setAttribute('role', 'img'); root.setAttribute('aria-label', title);
    if (pills) for (const label of root.querySelectorAll('.edgeLabel')) {
      const r = label.querySelector('rect'); if (!r) continue;
      // Padding is decorative and applied after layout; review edge-label collisions on upgrades.
      for (const [attr, delta] of [['x', -7], ['y', -3], ['width', 14], ['height', 6]])
        r.setAttribute(attr, Number(r.getAttribute(attr)) + delta);
    }
    const width = root.viewBox.baseVal.width;
    root.style.width = `${width}px`; root.style.maxWidth = 'none'; root.style.height = 'auto';
    const shadow = target.shadowRoot || target.attachShadow({ mode: 'open' });
    shadow.replaceChildren();
    const style = document.createElement('style');
    style.textContent = ':host{display:block;width:max-content;min-width:100%;font-size:15px;line-height:normal}svg{display:block;margin:0 auto}';
    shadow.append(style, root); return root;
  }

  const dialog = document.createElement('dialog'); dialog.className = 'fieldnotes-diagram-dialog';
  dialog.setAttribute('aria-labelledby', 'fieldnotes-diagram-title');
  const bar = document.createElement('div'); bar.className = 'fieldnotes-diagram-toolbar';
  const title = document.createElement('h2'); title.id = 'fieldnotes-diagram-title';
  const view = document.createElement('div'); view.className = 'fieldnotes-diagram-viewport';
  view.tabIndex = 0; view.setAttribute('role', 'region'); view.setAttribute('aria-label', 'Diagram. Scroll to explore.');
  const viewer = document.createElement('div'); view.append(viewer);
  let current, opener;
  function size(fit) {
    const svg = viewer.shadowRoot?.querySelector('svg'); if (!svg) return;
    const box = svg.viewBox.baseVal;
    const scale = fit ? Math.min(1, (view.clientWidth - 32) / box.width, (view.clientHeight - 32) / box.height) : 1;
    svg.style.width = `${box.width * scale}px`;
    viewer.style.minWidth = '100%';
    view.scrollTo(Math.max(0, (view.scrollWidth - view.clientWidth) / 2), 0);
  }
  bar.append(title, button('Fit', () => size(true)), button('100%', () => size(false)), button('Close', () => dialog.close()));
  dialog.append(bar, view); document.body.append(dialog);
  dialog.addEventListener('close', () => { current = undefined; opener?.focus(); });

  function expand(item, trigger) {
    current = item; opener = trigger; title.textContent = item.title;
    mount(viewer, item.svg, item.title, item.flow); dialog.showModal(); size(false);
  }

  function discover() {
    for (const pre of document.querySelectorAll('pre.fieldnotes-mermaid')) {
      if (diagrams.has(pre)) continue;
      const source = pre.textContent;
      const heading = [...document.querySelectorAll('h1,h2,h3')].filter(h => h.compareDocumentPosition(pre) & Node.DOCUMENT_POSITION_FOLLOWING).at(-1);
      const name = heading?.textContent.replace(/¶$/, '').trim() || 'Diagram';
      const figure = document.createElement('figure'); figure.className = 'fieldnotes-diagram';
      const toolbar = document.createElement('div'); toolbar.className = 'fieldnotes-diagram-toolbar';
      const status = document.createElement('span'); status.className = 'fieldnotes-diagram-status'; status.setAttribute('role', 'status');
      const canvas = document.createElement('div'); canvas.className = 'fieldnotes-diagram-viewport';
      canvas.tabIndex = 0; canvas.setAttribute('role', 'region'); canvas.setAttribute('aria-label', `${name}. Scroll to explore.`);
      const host = document.createElement('div'); canvas.append(host);
      const details = document.createElement('details'), summary = document.createElement('summary');
      summary.textContent = 'Diagram source'; details.append(summary);
      const item = { source, title: name, figure, host, status, details, svg: null };
      const expandButton = button('Expand diagram', () => expand(item, expandButton)); expandButton.disabled = true;
      item.expand = expandButton; toolbar.append(status, expandButton);
      pre.before(figure); details.append(pre); figure.append(toolbar, canvas, details); diagrams.set(pre, item);
    }
  }

  async function refresh() {
    pending = true; if (running) return; running = true;
    try {
      while (pending) {
        pending = false; discover();
        for (const [pre, item] of diagrams) {
          if (!pre.isConnected) { diagrams.delete(pre); continue; }
          item.status.textContent = 'Rendering diagram…'; item.expand.disabled = true;
          try {
            const mermaid = await load();
            // Let Mermaid detect diagram type: comments/frontmatter need not precede flowchart literally.
            mermaid.initialize(configuration(false));
            const parsed = await mermaid.parse(item.source);
            item.flow = ['flowchart', 'flowchart-v2', 'flowchart-elk'].includes(parsed.diagramType);
            mermaid.initialize(configuration(item.flow));
            const result = await mermaid.render(`fieldnotes-diagram-${serial++}`, item.source);
            if (!pre.isConnected) continue;
            item.svg = result.svg; mount(item.host, item.svg, item.title, item.flow);
            const viewport = item.host.parentElement;
            viewport.scrollTo(Math.max(0, (viewport.scrollWidth - viewport.clientWidth) / 2), 0);
            item.status.textContent = 'Scroll to explore'; item.expand.disabled = false;
            item.figure.dataset.renderState = 'ready';
            if (current === item && dialog.open) { mount(viewer, item.svg, item.title, item.flow); size(false); }
          } catch (error) {
            item.host.shadowRoot?.replaceChildren(); item.svg = null;
            item.status.textContent = 'Diagram unavailable. Source is shown below.';
            item.details.open = true; item.figure.dataset.renderState = 'error';
            console.error('Mermaid diagram rendering failed', error);
          }
        }
      }
    } finally { running = false; }
  }
  new MutationObserver(refresh).observe(document.body, { attributes: true, attributeFilter: ['data-md-color-scheme'] });
  if (typeof document$ !== 'undefined') document$.subscribe(() => { if (dialog.open) dialog.close(); refresh(); });
  else refresh();
})();
