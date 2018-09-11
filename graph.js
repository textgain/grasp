// graph.js v1.0.
// (c) Textgain

Math.clamp = function(v, min, max) {
	return Math.max(min, Math.min(max, v));
};

Point = function(x, y) {
	this.x = x;
	this.y = y;
	this.v = { // velocity
		x: 0,
		y: 0
	}
};

Point.random = function() {
	return new Point(
		Math.random(), 
		Math.random()
	);
};

bounds = function(p) {
	var x0 = +Infinity;
	var y0 = +Infinity;
	var x1 = -Infinity;
	var y1 = -Infinity;
	for (var i in p) {
		x0 = Math.min(x0, p[i].x);
		y0 = Math.min(y0, p[i].y);
		x1 = Math.max(x1, p[i].x);
		y1 = Math.max(y1, p[i].y);
	}
	return {x: x0, y: y0, width: x1 - x0, height: y1 - y0};
};

Graph = function(adj) {
	this.nodes = {};  // {node: Point}
	this.edges = adj; // {node1: {node2: length}}
	
	for (var n1 in this.edges) {
		for (var n2 in this.edges[n1]) {
			this.nodes[n1] = Point.random();
			this.nodes[n2] = Point.random();
		}
	}
};

Graph.default = {
	directed    : false,
	font        : '10px sans-serif',
	fill        : '#fff',
	stroke      : '#000',
	strokewidth : 0.5,
	radius      : 4.0,   // node radius
	
	f1          : 10.0,  // force constant (repulsion)
	f2          : 0.5,   // force constant (attraction)
	m           : 0.25   // force multiplier
};

Graph.default.update = function(obj) {
	var o = {};
	for (var k in this) {
		o[k] = (k in obj)? obj[k] : this[k];
	}
	return o;
};
	
Graph.prototype.update = function(options) {
	/* Updates node positions using a force-directed layout,
	 * where nodes repulse nodes (f1) and edges attract (f2).
	 */
	var o = Graph.default.update(options || {}); // {f1: 10.0, f2: 0.5, m: 0.25}
	var n = Object.keys(this.nodes);
	for (var i=0; i < n.length; i++) {
		for (var j=i+1; j < n.length; j++) {
			
			var n1 = n[i];
			var n2 = n[j];
			var p1 = this.nodes[n1];
			var p2 = this.nodes[n2];
			var dx = p1.x - p2.x;
			var dy = p1.y - p2.y;
			var d  = dx * dx + dy * dy + 0.1; // squared distance
			var f;
			
			// Repulsion (Coulomb's law).
			f = 0;
			if (d < 10000) // 100 * 100
				f = o.f1 * o.f1 / d;
			p1.v.x += dx * f;
			p1.v.y += dy * f;
			p2.v.x -= dx * f;
			p2.v.y -= dy * f;
			
			// Attraction (Hooke's law).
			f = 0;
			if (n1 in this.edges && n2 in this.edges[n1])
				f = o.f2 * o.f2 / Math.max(0.1, this.edges[n1][n2]); // n1 -> n2
			if (n2 in this.edges && n1 in this.edges[n2])
				f = o.f2 * o.f2 / Math.max(0.1, this.edges[n2][n1]); // n1 <- n2
			p1.v.x -= dx * f;
			p1.v.y -= dy * f;
			p2.v.x += dx * f;
			p2.v.y += dy * f;
		}
	}
	for (var n in this.nodes) {
		var p = this.nodes[n];
		p.x += Math.clamp(p.v.x * o.m, -10, +10);
		p.y += Math.clamp(p.v.y * o.m, -10, +10);
		p.v.x = 0;
		p.v.y = 0;
	}
};

Graph.prototype.render = function(ctx, x, y, options) {
	/* Draws the graph in the given <canvas> 2D context,
	 * representing nodes as circles and edges as lines.
	 */
	var o = Graph.default.update(options || {});
	var r = o.radius;
	var b = bounds(this.nodes);
	
	// Draw edges:
	ctx.save();
	ctx.translate(x - b.x - b.width / 2, y - b.y - b.height / 2);
	ctx.beginPath();
	for (var n1 in this.edges) {
		for (var n2 in this.edges[n1]) {
			var p1 = this.nodes[n1];
			var p2 = this.nodes[n2];
			ctx.moveTo(p1.x, p1.y);
			ctx.lineTo(p2.x, p2.y);
			
			// arrowhead
			if (o.directed) {
				var a = 3.14 + Math.atan2(p1.x - p2.x, p1.y - p2.y); // angle
				var x = p2.x - Math.sin(a) * r;
				var y = p2.y - Math.cos(a) * r;
				ctx.moveTo(
					x - (5.0 * Math.sin(a - 0.52)), // 30° = PI/6
					y - (5.0 * Math.cos(a - 0.52)));
				ctx.lineTo(x, y);
				ctx.lineTo(
					x - (5.0 * Math.sin(a + 0.52)), 
					y - (5.0 * Math.cos(a + 0.52)));
			}
		}
	}
	ctx.lineWidth = o.strokewidth;
	ctx.fillStyle = o.stroke;
	ctx.fill();
	ctx.strokeStyle = o.stroke;
	ctx.stroke();

	// Draw nodes:
	ctx.beginPath();
	for (var n in this.nodes) {
		var p = this.nodes[n];
		ctx.moveTo(p.x + r, p.y);
		ctx.arc(p.x, p.y, r, 0, 2 * Math.PI);
	}
	ctx.lineWidth = o.strokewidth * 2;
	ctx.fillStyle = o.fill;
	ctx.fill();
	ctx.stroke();

	// Draw node labels:
	ctx.font = o.font;
	for (var n in this.nodes) {
		var p = this.nodes[n];
		var s = String(n);
		ctx.fillStyle = o.stroke;
		ctx.fillText(s, p.x + r, p.y - r - 1);
	}
	ctx.restore();
};

Graph.prototype.animate = function(canvas, n, options) {
	/* Draws the graph in the given <canvas> element,
	 * iteratively updating the layout for n frames.
	 */
	function f() {
		if (n-- > 0) {
			var w = canvas.width;
			var h = canvas.height;
			var c = canvas.getContext('2d');
			c.clearRect(0, 0, w, h);
			this.update(options);
			this.update(options);
			this.render(c, w/2, h/2, options);
			window.requestAnimationFrame(f);
		}
	}
	f = f.bind(this);
	f();
};