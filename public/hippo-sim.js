let cols, rows;
let grid = [];
let openSet = [];
let closedSet = [];
let start, end;
let w = 30;
let path = [];
let hippo;
let hippoImg;
let moveSpeed = 10;
let moveCounter = 0;
let pathCompleted = false;

function preload() {
  hippoImg = loadImage('assets/hippo.png');
}

function setup() {
  createCanvas(600, 600);
  cols = floor(width / w);
  rows = floor(height / w);

  for (let i = 0; i < cols; i++) {
    grid[i] = [];
    for (let j = 0; j < rows; j++) {
      grid[i][j] = new Cell(i, j);
    }
  }

  start = grid[0][0];
  end = grid[cols - 1][rows - 1];
  start.wall = false;
  end.wall = false;

  openSet.push(start);

  hippo = {
    idx: 0,
    moving: false,
    x: start.i * w,
    y: start.j * w,
    bobbing: 0
  };
}

function draw() {
  background(220);

  if (openSet.length > 0 && !hippo.moving) {
    let lowest = 0;
    for (let i = 0; i < openSet.length; i++) {
      if (openSet[i].hippoCost < openSet[lowest].hippoCost) {
        lowest = i;
      }
    }

    let current = openSet[lowest];

    if (current === end) {
      hippo.moving = true;
    } else {
      removeFromArray(openSet, current);
      closedSet.push(current);

      let neighbors = current.getNeighbors();
      for (let neighbor of neighbors) {
        if (!closedSet.includes(neighbor) && !neighbor.wall) {
          let tempHippoCost = current.hippoCost + neighbor.cost;

          if (!openSet.includes(neighbor) || tempHippoCost < neighbor.hippoCost) {
            neighbor.hippoCost = tempHippoCost;
            neighbor.previous = current;

            if (!openSet.includes(neighbor)) {
              openSet.push(neighbor);
            }
          }
        }
      }
    }

    path = [];
    let temp = current;
    path.push(temp);
    while (temp.previous) {
      path.push(temp.previous);
      temp = temp.previous;
    }
  }

  drawGrid();
  drawHippo();
}

function drawGrid() {
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < rows; j++) {
      grid[i][j].show();
    }
  }
  for (let c of closedSet) {
    c.highlight(color(255, 100, 100, 180));
  }
  for (let o of openSet) {
    o.highlight(color(100, 255, 100, 180));
  }
  for (let p of path) {
    p.highlight(color(0, 0, 255, 100));
  }
}

function drawHippo() {
  if (hippo.moving && !pathCompleted) {
    if (moveCounter % moveSpeed === 0 && hippo.idx < path.length) {
      let p = path[path.length - 1 - hippo.idx];
      hippo.x = p.i * w + w / 2;
      hippo.y = p.j * w + w / 2;
      hippo.idx++;
      if (hippo.idx >= path.length) {
        pathCompleted = true;
      }
    }
    moveCounter++;
  }

  if (hippo.moving || pathCompleted) {
    hippo.bobbing = sin(frameCount * 0.1) * 1.5;
    noStroke();
    fill(0, 0, 0, 50);
    ellipse(hippo.x, hippo.y + 5, w * 0.7, w * 0.3);
    imageMode(CENTER);
    image(hippoImg, hippo.x, hippo.y + hippo.bobbing, w * 1.1, w * 1.1);
  }
}

function removeFromArray(arr, elt) {
  let idx = arr.indexOf(elt);
  if (idx > -1) {
    arr.splice(idx, 1);
  }
}

class Cell {
  constructor(i, j) {
    this.i = i;
    this.j = j;
    this.wall = random(1) < 0.15;
    this.cost = this.wall ? 50 : random(1, 10);
    this.hippoCost = Infinity;
    this.previous = undefined;
  }

  getNeighbors() {
    let neighbors = [];
    if (this.i < cols - 1) neighbors.push(grid[this.i + 1][this.j]);
    if (this.i > 0) neighbors.push(grid[this.i - 1][this.j]);
    if (this.j < rows - 1) neighbors.push(grid[this.i][this.j + 1]);
    if (this.j > 0) neighbors.push(grid[this.i][this.j - 1]);
    return neighbors;
  }

  show() {
    noStroke();
    if (this.wall) {
      fill(139, 69, 19);
    } else {
      fill(0, 100, 255, map(this.cost, 1, 10, 50, 200));
    }
    rect(this.i * w, this.j * w, w - 1, w - 1);
  }

  highlight(col) {
    fill(col);
    rect(this.i * w, this.j * w, w - 1, w - 1);
  }
}
