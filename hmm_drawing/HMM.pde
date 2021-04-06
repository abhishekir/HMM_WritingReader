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
