import streamlit.components.v1 as components

def inject_micro_interactions():
    """
    Injects Anime.js micro-interactions into the Streamlit parent DOM.
    Designed to be robust against Streamlit's frequent re-renders by tracking
    animated elements with data attributes and injecting the library only once.
    """
    js_code = """
    <script>
        const pWin = window.parent;
        const pDoc = pWin.document;

        function runAnimations() {
            if (typeof pWin.anime === 'undefined') return;
            const anime = pWin.anime;

            // 1. Message bubbles (handoff items)
            const handoffs = pDoc.querySelectorAll('.handoff-item:not([data-animated])');
            if (handoffs.length > 0) {
                handoffs.forEach(el => el.setAttribute('data-animated', 'true'));
                anime({
                    targets: handoffs,
                    translateY: [20, 0],
                    opacity: [0, 1],
                    delay: anime.stagger(100),
                    duration: 600,
                    easing: 'easeOutCubic'
                });
            }

            // 2. Metric cards (gauges / datum cells)
            const gauges = pDoc.querySelectorAll('.gauge:not([data-animated]), .datum .cell:not([data-animated])');
            if (gauges.length > 0) {
                gauges.forEach(el => el.setAttribute('data-animated', 'true'));
                anime({
                    targets: gauges,
                    scale: [0.9, 1],
                    opacity: [0, 1],
                    delay: anime.stagger(80),
                    duration: 500,
                    easing: 'easeOutBack'
                });
            }

            // 3. Hover float effect for charts
            const charts = pDoc.querySelectorAll('[data-testid="stVegaLiteChart"]:not([data-floating])');
            if (charts.length > 0) {
                charts.forEach(el => {
                    el.setAttribute('data-floating', 'true');
                    let floatAnim;
                    el.addEventListener('mouseenter', () => {
                        floatAnim = anime({
                            targets: el,
                            translateY: -4,
                            direction: 'alternate',
                            loop: true,
                            duration: 1200,
                            easing: 'easeInOutSine'
                        });
                    });
                    el.addEventListener('mouseleave', () => {
                        if (floatAnim) floatAnim.pause();
                        anime({
                            targets: el,
                            translateY: 0,
                            duration: 300,
                            easing: 'easeOutQuad'
                        });
                    });
                });
            }
        }

        if (!pWin.__anime_script_loaded) {
            pWin.__anime_script_loaded = true;
            const script = pDoc.createElement('script');
            script.src = "https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.1/anime.min.js";
            script.onload = () => {
                pWin.__anime_ready = true;
                runAnimations();
                
                // Add a MutationObserver to catch elements rendered slightly after the script runs
                const observer = new MutationObserver(() => {
                    runAnimations();
                });
                observer.observe(pDoc.body, { childList: true, subtree: true });
            };
            pDoc.head.appendChild(script);
        } else {
            if (pWin.__anime_ready) {
                runAnimations();
            }
        }
    </script>
    """
    components.html(js_code, height=0)
