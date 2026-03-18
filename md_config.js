module.exports = {
  script: [
    {
      content: `
        window.MathJax = {
          tex: {
            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']]
          },
          svg: {
            fontCache: 'global'
          }
        };
      `
    },
    {
      url: 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js'
    }
  ],
  pdf_options: {
    format: 'Letter',
    margin: { top: '20mm', right: '20mm', bottom: '20mm', left: '20mm' },
    printBackground: true
  }
};
