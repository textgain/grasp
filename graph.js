// graph.js v3.0.
// (c) Textgain

// ------------------------------------------------------------------------------------------------

clamp = function(v, min=0.0, max=1.0) {
	return Math.min(Math.max(v, min), max);
};

Point = function(x, y) {
	this.x = isNaN(x) ? Math.random() : x;
	this.y = isNaN(y) ? Math.random() : y;
	this.v = { // velocity
		x: 0,
		y: 0
	};
};

bounds = function(p=[]) {
	let x0 = +Infinity;
	let y0 = +Infinity;
	let x1 = -Infinity;
	let y1 = -Infinity;
	for (let i in p) {
		x0 = Math.min(x0, p[i].x);
		y0 = Math.min(y0, p[i].y);
		x1 = Math.max(x1, p[i].x);
		y1 = Math.max(y1, p[i].y);
	}
	return {
		x: x0, 
		y: y0, 
		w: x1 - x0, 
		h: y1 - y0
	};
};

// ------------------------------------------------------------------------------------------------

Graph = function(adj={}) {
	this.nodes = {};  // {node: Point}
	this.edges = adj; // {node1: {node2: weight}}
	
	for (let n1 in this.edges) {
		for (let n2 in this.edges[n1]) {
			this.nodes[n1] = new Point();
			this.nodes[n2] = new Point();
		}
	}
};

Graph.prototype.add = function(n1, n2, weight=1.0) {
	if (!this.nodes[n1])
		 this.nodes[n1] = new Point();
	if (!this.nodes[n2])
		 this.nodes[n2] = new Point();
	if (!this.edges[n1])
		 this.edges[n1] = {}
	if (!this.edges[n1][n2])
		 this.edges[n1][n2] = weight;
};

// ------------------------------------------------------------------------------------------------

Graph.default = {
	callback    : false,  // function(graph, iteration)
	directed    : false,
	font        : '10px sans-serif',
	fill        : '#fff', // node color
	stroke      : '#000', // edge color
	strokewidth : 0.5,    // edge width
	radius      : 4.0,    // node radius
	
	k1          : 1.0,    // force constant (repulse)
	k2          : 1.0,    // force constant (attract)
	k           : 1.0,    // force constant (low = compact)
	m           : 1.0     // force dampener (low = smooth)
};

// ------------------------------------------------------------------------------------------------

Graph.prototype.update = function(options={}) {
	/* Updates node positions using a force-directed layout,
	 * where nodes repulse nodes (f1) and edges attract (f2).
	 */
	let o = Object.assign(Graph.default, options);
	let n = Object.keys(this.nodes);
	
	for (let i=0; i < n.length; i++) {
		for (let j=i+1; j < n.length; j++) {
			let n1 = n[i];
			let n2 = n[j];
			let p1 = this.nodes[n1];
			let p2 = this.nodes[n2];
			let dx = p1.x - p2.x;
			let dy = p1.y - p2.y;
			let d2 = dx * dx + dy * dy;
			let d  = d2 ** 0.5; // distance
			let k  = o.k * 1e+2;
			let m  = o.m * 1e-1;
			let f1 = 0;
			let f2 = 0;
			
			// Repulse nodes (Coulomb's law)
			if (d < k * 10)
				f1 = o.k1 * k ** 2 / d2 / 20;
			
			// Attract nodes (Hooke's law)
			if ((n1 in this.edges && n2 in this.edges[n1]) ||
				(n2 in this.edges && n1 in this.edges[n2]))
				f2 = o.k2 * (d2 - k ** 2) / k / d;
			
			p1.v.x += dx * (f1 - f2) * m;
			p1.v.y += dy * (f1 - f2) * m;
			p2.v.x -= dx * (f1 - f2) * m;
			p2.v.y -= dy * (f1 - f2) * m;
		}
	}
	for (let n in this.nodes) {
		let p = this.nodes[n];
		p.x += clamp(p.v.x, -10, +10);
		p.y += clamp(p.v.y, -10, +10);
		p.v.x = 0;
		p.v.y = 0;
	}
};

Graph.prototype.render = function(ctx, x, y, options={}) {
	/* Draws the graph in the given <canvas> 2D context,
	 * representing nodes as circles and edges as lines.
	 */
	let o = Object.assign(Graph.default, options);
	let r = o.radius;
	let b = bounds(this.nodes);
	
	ctx.save();
	ctx.translate(
		x - b.x - b.w / 2, // center
		y - b.y - b.h / 2
	);
	
	// Draw edges:
	ctx.beginPath();
	let seen = new Set();
	for (let n1 in this.edges) {
		for (let n2 in this.edges[n1]) {
			let p1 = this.nodes[n1];
			let p2 = this.nodes[n2];
			
			if (!(n2 in this.edges && 
				  n1 in this.edges[n2] && seen.has(n2))) {
				ctx.moveTo(p1.x, p1.y);
				ctx.lineTo(p2.x, p2.y);
			}
			
			// arrowheads?
			if (o.directed) {
				let a  = 3.14 + Math.atan2(p1.x - p2.x, p1.y - p2.y); // direction
				let x1 = p2.x - Math.sin(a) * r;
				let y1 = p2.y - Math.cos(a) * r;
				let x2 = x1 - ( Math.sin(a - 0.5) * 5); // 0.5 ≈ 30°
				let y2 = y1 - ( Math.cos(a - 0.5) * 5);
				let x3 = x1 - ( Math.sin(a + 0.5) * 5);
				let y3 = y1 - ( Math.cos(a + 0.5) * 5);
				ctx.moveTo(x1, y1);
				ctx.lineTo(x2, y2);
				ctx.lineTo(x3, y3);
			}
			seen.add(n1);
		}
	}
	ctx.lineWidth = o.strokewidth;
	ctx.fillStyle = o.stroke;
	ctx.fill();
	ctx.strokeStyle = o.stroke;
	ctx.stroke();

	// Draw nodes:
	ctx.beginPath();
	for (let n in this.nodes) {
		let p = this.nodes[n];
		ctx.moveTo(p.x + r, p.y);
		ctx.arc(p.x, p.y, r, 0, 2 * Math.PI);
	}
	ctx.lineWidth = o.strokewidth * 2;
	ctx.fillStyle = o.fill;
	ctx.fill();
	ctx.stroke();

	// Draw labels:
	ctx.font = o.font;
	for (let n in this.nodes) {
		let p = this.nodes[n];
		let s = new String(n);
		ctx.fillStyle = o.stroke;
		ctx.fillText(s, p.x + r, p.y - r - 1);
	}
	ctx.restore();
};

Graph.prototype.animate = function(canvas, n, options={}) {
	/* Draws the graph in the given <canvas> element,
	 * iteratively updating the layout for n frames.
	 */
	let i = 0;
	function f() {
		if (i++ < n) {
			let w = canvas.width;
			let h = canvas.height;
			let g = canvas.getContext('2d');
			g.clearRect(0, 0, w, h);
			this.canvas = canvas;
			this.update(options);
			this.update(options);
			this.render(g, w/2, h/2, options);
			window.requestAnimationFrame(f);
		}
		if (options.callback) {
			options.callback(this, i);
		}
	}
	f = f.bind(this);
	f();
};

// ------------------------------------------------------------------------------------------------

Graph.prototype.centrality = function() { 
	/* Returns a {node: weight} (0.0-1.0),
	 * based on Google PageRank algorithm.
	 */
	let len = (o) => Object.keys(o).length;

	let n = this.nodes;
	let e = this.edges;
	let w = {};
	for (let k in n)
		w[k] = 1 / len(n);
	for (let i=0; i<100; i++) {
		let p = {...w};
		for (let k1 in n) {
			for (let k2 in e[k1] || {})
				w[k2] += 0.85 * p[k1] * e[k1][k2] / len(e[k1]);
			w[k1] += 1 - 0.85;
		}
		let m = 0; // normalize
		let ε = 0; // converged?
		for (k in n)
			m += w[k] ** 2.0;
		for (k in n)
			w[k] /= m ** 0.5 || 1;
		for (k in n)
			ε += Math.abs(w[k] - p[k]);
		if (ε <= len(n) * 0.00001)
			break;
	}
	return w;
}

// ------------------------------------------------------------------------------------------------

capture = function(canvas, n=1) {
	/* Downloads a JSON file with n frames,
	 * where every frame is a PNG data URI.
	 */
	let a = [];
	let i = 0;
	let e;
	function f() {
		if (i++ < n) {
			a.push(canvas.toDataURL());
			window.requestAnimationFrame(f);
		} else {
			a = JSON.stringify(a)
			e = document.createElement('a');
			e.download = 'frames.json';
			e.href = 'data:text/plain,' + a;
			e.click();
		}
	}
	f();
}

// for i, s in enumerate(json.load(open('frames.json'))):
//     s = s.split(',')[1]
//     s = base64.b64decode(s)
//     f = open('frame%i.png' % i, 'wb')
//     f.write(s)
//     f.close()

// ------------------------------------------------------------------------------------------------

//	<canvas id="g" width=640 height=480></canvas>
//	<script src="graph.js"></script>
//	<script>
//		var adjacency = {
//			'node1': {'node2': 1.0},
//			'node2': {'node3': 1.0}
//		};
//		var canvas;
//		canvas = document.getElementById("g");
//		canvas.graph = new Graph(adjacency);
//		canvas.graph.animate(canvas, 1000, {
//			directed    : true,
//			fill        : '#fff',
//			stroke      : '#000',
//			strokewidth : 0.5,
//			radius      : 4.0,
//		});
//	</script>
