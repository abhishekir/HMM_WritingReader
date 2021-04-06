static final float EPSILON = 0.0000001;

ArrayList<InkBlot> blots;
int lastMove;
PVector lastLoc;
Boolean penDown;
float lastX, lastY;
ModelCollection models;
Character prediction;

float MESSAGE_X = 20;
float MESSAGE_Y = 20;

static final int UP = 0;
static final int UPRIGHT = 1;
static final int RIGHT = 2;
static final int DOWNRIGHT = 3;
static final int DOWN = 4;
static final int DOWNLEFT = 5;
static final int LEFT = 6;
static final int UPLEFT = 7;
static final int STILL = 8;
static final int PENUP = 9;
static final int PENDOWN = 10;

static final int POSSIBLE_MOVES = 11;

static PVector[] dirVecs = {
  new PVector(0,-1), (new PVector(1,-1)).normalize(),
  new PVector(1,0), (new PVector(1,1)).normalize(),
  new PVector(0,1), (new PVector(-1,1)).normalize(),
  new PVector(-1,0), (new PVector(-1,-1)).normalize()
};

void setup() {
  size(400,400);
  blots = new ArrayList<InkBlot>();
  lastMove = PENUP;
  penDown = false;
  models = new ModelCollection();
  prediction = ' ';
}

class InkBlot {
  float x;
  float y;
  
  InkBlot(float x, float y) {
    this.x = x;
    this.y = y;
  }
  
  void draw() {
    fill(0);
    ellipse(x,y,20,20);
  }
}

// Basic matrix vector mult; allocates new matrix
double[] multByVec(double[][] A, double[] B) {
  if (A[0].length != B.length) {
    System.err.print("Illegal matrix multiplication; ");
    System.err.println("a x b can only be mult by length b vector");
    System.exit(1);
  }
  // Autoinitialize arrays to 0 in Java (this is not something all languages do)
  double[] out = new double[A.length];
  for (int i = 0; i < A.length; i++) {
    for (int j = 0; j < A[0].length; j++) {
        out[i] += A[i][j] * B[j];
    }
  }
  return out;
}

int getDirCode(float deltaX, float deltaY) {
  if (abs(deltaX) < EPSILON && abs(deltaY) < EPSILON) {
    return STILL;
  }
  // Probably the least annoying way to do this is a dot product
  // with each potential vector, returning the one with largest cosine
  // and thus best matching angle.  This would scale best if we increase
  // angle discretization resolution (instead of using fixed thresholds, for example).
  PVector newVector = new PVector(deltaX, deltaY);
  newVector.normalize();
  int bestMove = STILL;
  double bestDot = -1;
  for (int i = 0; i < dirVecs.length; i++) {
    double myDot = newVector.dot(dirVecs[i]);
    if (myDot > bestDot) {
      bestMove = i;  // bestVecs order matches the constant names
      bestDot = myDot;
    }
  }
  return bestMove;
}

void draw() {
  background(128,128,128);
  boolean training = false;
  
  if (keyPressed && (key == BACKSPACE || key == DELETE)) {
    blots = new ArrayList<InkBlot>();
    models.resetBeliefs();
    prediction = ' ';
  } else if (keyPressed && Character.isLetterOrDigit(key)) {
    // Holding down a key indicates that we're training that letter
    fill(255);
    prediction = ' ';
    text("Training " + key + " model...", MESSAGE_X, MESSAGE_Y);
    training = true;
  }
    
  if (mousePressed) {
    InkBlot newBlot = new InkBlot(mouseX,mouseY);
    blots.add(newBlot);
    if (lastMove == PENUP) {
      lastMove = PENDOWN;
      lastX = mouseX;
      lastY = mouseY;
    } else {
      int move = getDirCode(mouseX - lastX, mouseY - lastY);
      if (training) {
          models.appendData(key, move);
          // This could get a bit inefficient if we were using Forward-Backward,
          // but should be fine here
          models.updateModel(key);
      } else {
        // No key held means we're guessing a letter
        models.updateBeliefs(move);
        prediction = models.getPrediction();
      }
      lastMove = move;
      lastX = mouseX;
      lastY = mouseY;
    }
      
  } else {
    lastMove = PENUP;
  }
  
  if (prediction != ' ') {
    text("Prediction: " + prediction, MESSAGE_X, MESSAGE_Y);
  }
  
  for (InkBlot o : blots) {
    o.draw();
  }
}
