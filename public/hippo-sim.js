let simSketch = (p) => {
    let cols, rows;
    let grid = [];
    let openSet = [];
    let closedSet = [];
    let start, end;
    let w;
    let path = [];
    let hippo;
    let hippoImg;
    let moveSpeed = 10;
    let moveCounter = 0;
    let pathCompleted = false;
    let gridSize;
    let difficulty;
    let stepCount = 0;
  
    p.preload = () => {
      hippoImg = p.loadImage('assets/hippo.png');
    };
  
    p.setup = () => {
      gridSize = parseInt(document.getElementById("gridSize").value);
      difficulty = document.getElementById("difficulty").value;
      updateDifficultyLabel();
  
      w = p.floor(600 / gridSize);
      canvasEl = p.createCanvas(600, 600).parent("sketch-holder");
      cols = p.floor(p.width / w);
      rows = p.floor(p.height / w);
  
      grid = [];
      openSet = [];
      closedSet = [];
      path = [];
      moveCounter = 0;
      stepCount = 0;
      pathCompleted = false;
  
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
  
      document.getElementById("grid-size").innerText = `${cols}x${rows}`;
      updateStats();
    };
  
    p.draw = () => {
      p.background(220);
  
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
    };
  
    function drawGrid() {
      for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
          grid[i][j].show(p);
        }
      }
      for (let c of closedSet) c.highlight(p.color(255, 100, 100, 180), p);
      for (let o of openSet) o.highlight(p.color(100, 255, 100, 180), p);
      for (let pt of path) pt.highlight(p.color(0, 0, 255, 100), p);
    }
  
    function drawHippo() {
      if (hippo.moving && !pathCompleted) {
        if (moveCounter % moveSpeed === 0 && hippo.idx < path.length) {
          let pCell = path[path.length - 1 - hippo.idx];
          hippo.x = pCell.i * w + w / 2;
          hippo.y = pCell.j * w + w / 2;
          hippo.idx++;
          stepCount++;
          updateStats();
          if (hippo.idx >= path.length) pathCompleted = true;
        }
        moveCounter++;
      }
  
      if (hippo.moving || pathCompleted) {
        hippo.bobbing = p.sin(p.frameCount * 0.1) * 1.5;
        p.noStroke();
        p.fill(0, 0, 0, 50);
        p.ellipse(hippo.x, hippo.y + 5, w * 0.7, w * 0.3);
        p.imageMode(p.CENTER);
        p.image(hippoImg, hippo.x, hippo.y + hippo.bobbing, w * 1.1, w * 1.1);
      }
    }
  
    function removeFromArray(arr, elt) {
      let idx = arr.indexOf(elt);
      if (idx > -1) arr.splice(idx, 1);
    }
  
    function updateStats() {
      document.getElementById("step-count").innerText = stepCount;
    }
  
    function updateDifficultyLabel() {
      document.getElementById("difficulty-label").innerText =
        difficulty.charAt(0).toUpperCase() + difficulty.slice(1);
    }
  
    class Cell {
      constructor(i, j) {
        this.i = i;
        this.j = j;
        this.wall = p.random(1) < getWallProbability();
        this.cost = this.wall ? 50 : p.random(1, 10);
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
  
      show(p) {
        p.noStroke();
        if (this.wall) {
          p.fill(139, 69, 19);
        } else {
          p.fill(0, 100, 255, p.map(this.cost, 1, 10, 50, 200));
        }
        p.rect(this.i * w, this.j * w, w - 1, w - 1);
      }
  
      highlight(col, p) {
        p.fill(col);
        p.rect(this.i * w, this.j * w, w - 1, w - 1);
      }
    }
  
    function getWallProbability() {
      switch (difficulty) {
        case "easy": return 0.05;
        case "medium": return 0.15;
        case "hard": return 0.25;
        default: return 0.15;
      }
    }
  };
  
  // Start the sketch
  new p5(simSketch, document.getElementById("sketch-holder"));
  
  // Remove function to stop sketch
  function removeSketch() {
    if (canvasEl && canvasEl.remove) canvasEl.remove();
  }
  