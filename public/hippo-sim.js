let hippos = [];
let predators = [];
let food;
let w = 600;
let h = 600;
let numHippos = 10;
let numPredators = 5;
let lambda0 = 0.9;
let c = 0.02;
let t = 0;
let lambda;

function setup() {
  let canvas = createCanvas(w, h);
  canvas.parent("sketch-holder");
  lambda = lambda0;

  for (let i = 0; i < numHippos; i++) {
    hippos.push({
      pos: createVector(random(width), random(height)),
      fitness: Infinity
    });
  }

  for (let i = 0; i < numPredators; i++) {
    predators.push(createVector(random(width), random(height)));
  }

  food = createVector(random(width), random(height));
}

function draw() {
  background(30, 144, 255, 100);
  fill(0, 255, 0);
  ellipse(food.x, food.y, 20, 20);

  for (let p of predators) {
    noStroke();
    fill(255, 0, 0, 50);
    ellipse(p.x, p.y, 80 + sin(frameCount * 0.1) * 10);
    fill(255, 0, 0);
    stroke(0);
    strokeWeight(1);
    ellipse(p.x, p.y, 30 + sin(frameCount * 0.1) * 5);
  }

  for (let h of hippos) {
    let r = random();

    if (r < lambda) {
      let bestPos = getBestHippo();
      let alpha = random(0.1, 0.5);
      let beta = random(0.1, 0.5);
      let awayFromPred = predatorAvoidance(h.pos);
      h.pos.add(p5.Vector.mult(p5.Vector.sub(bestPos, h.pos), alpha));
      h.pos.add(p5.Vector.mult(awayFromPred, beta));
    } else {
      let gamma = random(0.1, 0.3);
      h.pos.add(p5.Vector.mult(p5.Vector.sub(food, h.pos), gamma));
    }

    h.pos.x = constrain(h.pos.x, 0, width);
    h.pos.y = constrain(h.pos.y, 0, height);

    h.fitness = dist(h.pos.x, h.pos.y, food.x, food.y);
    fill(r < lambda ? 100 : 0);
    ellipse(h.pos.x, h.pos.y, 15, 15);
  }

  lambda = lambda0 * exp(-c * t);
  t++;
}

function getBestHippo() {
  let best = hippos[0];
  for (let h of hippos) {
    if (h.fitness < best.fitness) {
      best = h;
    }
  }
  return best.pos.copy();
}

function predatorAvoidance(pos) {
  let avoidance = createVector(0, 0);
  for (let p of predators) {
    let d = dist(pos.x, pos.y, p.x, p.y);
    if (d < 100) {
      let away = p5.Vector.sub(pos, p);
      away.normalize();
      avoidance.add(away);
    }
  }
  return avoidance;
}
