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

static PVector[] dirVecs =
  { new PVector(0,-1),
    (new PVector(1,-1)).normalize(),
    new PVector(1,0),
    (new PVector(1,1)).normalize(),
    new PVector(0,1),
    (new PVector(-1,1)).normalize(),
    new PVector(-1,0),
    (new PVector(-1,-1)).normalize()
  };
    
    

void setup() {
  size(400,400);
  blots = new ArrayList<InkBlot>();
  lastMove = PENUP;
  penDown = false;
  models = new ModelCollection();
  prediction = ' ';
}

// Adapted from steering assignment obstacles
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

class HMM {
  Character name;
  double[][] T;
  double[][] O;
  ArrayList<Integer> trainingData;
  double[] hiddenStateProbs;
  double logProb;
  
  HMM(Character name) {
    this.name = name;
    this.trainingData = new ArrayList<Integer>();
    reset_beliefs();
  }
  
  // addData:  Keep retraining separate, since it could be expensive depending
  // on the method
  void addData(int move) {
    trainingData.add(move);
  }
  
  // Set the hidden state probs back to uniform and log prob to 0 (log of 1).
  // Doesn't get rid of training data.
  void reset_beliefs() {
    logProb = 0;
    hiddenStateProbs = new double[POSSIBLE_MOVES];
    for (int i = 0; i < hiddenStateProbs.length; i++) {
      hiddenStateProbs[i] = 0.01;
    }
    hiddenStateProbs[PENDOWN] = 0.9;
  }
  
  // simpleTrain:  Set the transition model to be exactly the transition probabilities
  // measured in the data; set observation probabilities to be mostly the true
  // observation, but some noise
  // For the simple model, we'll just make up some observation probabilities:
  // 70% intended direction
  // 15% Still pen
  // 4% Nearby directions (2)
  // 1% Other (7)
  
  void simpleTrain() {
    //INITIALIZING TRANSITION MATRIX
    T = new double[POSSIBLE_MOVES][POSSIBLE_MOVES];
    //populate transition matrix with 0
    for(int state = 0; state < POSSIBLE_MOVES; state ++) {
      for(int prev_state = 0; prev_state < POSSIBLE_MOVES; prev_state ++) {
        T[state][prev_state] = 0;
      }
    }
    
    //count all times each state transitioned to another state
    for(int i = 1; i < trainingData.size(); i++) {
      int state = trainingData.get(i);
      int prev_state = trainingData.get(i-1);
      T[state][prev_state] += 1;
    }
    
    //normalize to 1
    for(int prev_state = 0; prev_state < POSSIBLE_MOVES; prev_state ++) {
       //find total count of observed transitions for each state
       int tot_count = 0;
       for(int state = 0; state < POSSIBLE_MOVES; state ++) {
         tot_count += T[state][prev_state];
       }
       
       //normalize so columns sum to 1 (divide by total count)
       if(tot_count != 0) {
         for(int state = 0; state < POSSIBLE_MOVES; state ++) {
           T[state][prev_state] = T[state][prev_state]/tot_count; 
         }
       }
    }
    
    //INITIALIZING OBSERVATION MATRIX
    O = new double[POSSIBLE_MOVES][POSSIBLE_MOVES];
    //Where O[state][observation] is the probability that we are in 'state' (certain direction/movement) when
    //the observed state is 'observation' (also a direction/movement)
    for(int observation = 0; observation < POSSIBLE_MOVES; observation ++) {
      for(int state = 0; state < POSSIBLE_MOVES; state ++) {
        
        //initialize direction probabilites: .7 chance observed dir is true, .15 chance observed dir is actually STILL, .04 chance observed dir is one of the adjace directions, .01 chance all else
        if(observation < STILL) {
          if(observation == state) {
            O[state][observation] = .7; 
          }
          else if (state == STILL) {
            O[state][observation] = .15; 
          }
          //adjacent directions
          else if (state == (observation + 1) % 8 || state == (observation - 1) % 8) {
            O[state][observation] = .04;
          }
          else {
            O[state][observation] = .01; 
          }
        }
        
        //initialize STILL probabilities: .7 chance observed STILL is STILL, equal chance to all else
        else if(observation == STILL) {
           if(observation == state) {
             O[state][observation] = .7;
           }
           else {
             O[state][observation] = .3/(POSSIBLE_MOVES - 1);
           }
        }
        //initialize PENUP/PENDOWN probabilities: 1 chance observed PENUP/PENDOWN is actually PENUP/PENDOWN, 0 chance to all else
        else {
          if(observation == state) {
            O[state][observation] = 1; 
          }
          else {
            O[state][observation] = 0; 
          }
        }
      }
    }
  }

  // Forward algorithm
  //   * Multiply state likelihoods by probability of observation
  //   * Sum those, take log, and add this to log likelihood of the overall model
  //   * Normalize state likelihoods to sum to 1 - this is new state vector
  //   * Make prediction for next time step with multiplication by transition matrix
  // These are all just changing HMM state; no return value
  void forward(int obs) {
    for(int move = 0; move < POSSIBLE_MOVES; move ++) {
       double stateProb = hiddenStateProbs[move];
       //probability of hidden state (direction/movement) multiplied by likelihood that that hidden state (direction/move) is produced by observation obs (direction/move)
       stateProb = stateProb * (O[move][obs]);
       hiddenStateProbs[move] = stateProb;
    }
    
    //sum likelihoods calculated above
    float sumLikelihoods = 0;
    for(double stateProb : hiddenStateProbs) {
      sumLikelihoods += stateProb; 
    }
    
    float logLikelihood = log(sumLikelihoods);
    logProb += logLikelihood;
    
    //normalize hidden state vector
    for(int move = 0; move < POSSIBLE_MOVES; move ++) {
       double stateProb = hiddenStateProbs[move];
       stateProb = stateProb/sumLikelihoods;
       hiddenStateProbs[move] = stateProb;
    }
    
    //calculate new hidden state vector by multiplying calculated vector by transition matrix
    hiddenStateProbs = multByVec(T, hiddenStateProbs);
    
  }
  
  double getLogProb() {
    return logProb;
  }
  
}

class ModelCollection {
  
  HashMap<Character, HMM> myModels;
  
  ModelCollection() {
    myModels = new HashMap<Character, HMM>();
  }
  
  void appendData(char modelname, int move) {
    HMM model = myModels.get(modelname);
    if (model == null) {
      HMM newModel = new HMM(modelname);
      newModel.addData(move);
      myModels.put(modelname, newModel);
      for (HMM m : myModels.values()) {
        m.reset_beliefs(); // Reset log probabilities and hidden state vector
      }
    } else {
      model.addData(move);
    }
  }
  
  void updateModel(char modelname) {
    HMM model = myModels.get(modelname);
    if (model == null) {
      System.err.println("Couldn't find model: " + modelname);
      return;
    }
    model.simpleTrain();
  }
  
  void updateBeliefs(int move) {
    for (HMM model : myModels.values()) {
      model.forward(move);
    }
  }
  
  void resetBeliefs() {
    for (HMM model : myModels.values()) {
      model.reset_beliefs();
    }
  }
  
  Character getPrediction() {
    double bestLogProb = Double.NEGATIVE_INFINITY;
    Character bestModel = '?';
    // Leave this debug information - the graders will find it useful as well
    System.err.println("Log likelihoods: ");
    for (Character m : myModels.keySet()) {
      System.err.print(m + ":");
      HMM model = myModels.get(m);
      double logProb = model.getLogProb();
      System.err.println(logProb);
      if (logProb > bestLogProb) {
        bestLogProb = logProb;
        bestModel = m;
      }
    }
    return bestModel;
  }
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
