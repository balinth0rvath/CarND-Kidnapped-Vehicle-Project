#include <cmath>

/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   */
  num_particles = 15;  // TODO: Set the number of particles
	std::default_random_engine gen;
	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];
	std::normal_distribution<double> dist_x(x, std_x);
	std::normal_distribution<double> dist_y(y, std_y);
	std::normal_distribution<double> dist_theta(theta, std_theta);
	for (int i=0; i< num_particles; ++i)
	{
		Particle particle;
		// add random noise to positions
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		// initialize all weights to one
		particle.weight = 1.0;
		particles.push_back(particle);	
		weights.push_back(particle.weight);
	}
  is_initialized = true;	
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   */
	std::default_random_engine gen;
	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];

	for(int i=0; i<num_particles;++i)
	{
		double x0 = particles[i].x;
		double y0 = particles[i].y;
		double theta0 = particles[i].theta;
	
		// calculate the new postion and heading using the previous ones,
		// velocity, yawn rate. Use bicyle motion model	
		double x_pred = 0;
		double y_pred = 0;
		if (fabs(yaw_rate) < 0.000001)
		{
			x_pred = x0 + velocity * delta_t * cos(theta0);
			y_pred = y0 + velocity * delta_t * sin(theta0);
		} else
		{
			x_pred = x0 + (velocity / yaw_rate) * 
											(sin(theta0 +yaw_rate * delta_t) - sin(theta0));

			y_pred = y0 + (velocity / yaw_rate) * 
											(cos(theta0) - cos(theta0 +yaw_rate * delta_t));
		}
		double theta_pred = theta0 + yaw_rate * delta_t;
	
		// add noise
		std::normal_distribution<double> dist_x(x_pred, std_x);
		std::normal_distribution<double> dist_y(y_pred, std_y);
		std::normal_distribution<double> dist_theta(theta_pred, std_theta);
		
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   */
	for(unsigned int j=0; j<observations.size(); ++j)
	{
			int id = -1;
			// Calcukating an exact bound for lowest distance is not important
			// since landmarks were filtered to a range in a previous step 
			double lowest_distance = 999.9;
			
			for(unsigned int k=0; k<predicted.size(); ++k)
			{
				double distance = dist(observations[j].x, observations[j].y,
															predicted[k].x, predicted[k].y);
				if (distance < lowest_distance)
				{
					lowest_distance = distance;	
					id = predicted[k].id;	
				}
			}
			observations[j].id = id;
	}
}

vector<LandmarkObs> ParticleFilter::filterLandmarks(const Particle& particle, 
										 const Map &map_landmarks, double sensor_range)
{
  /**
   * Collect landmarks in sensor range respect to a specific particle 
	 * to improve speed
   */

	vector<LandmarkObs> nearby_landmarks;
	for(unsigned int j=0; j<map_landmarks.landmark_list.size(); ++j)
	{
		if (dist( map_landmarks.landmark_list[j].x_f,
							map_landmarks.landmark_list[j].y_f,
							particle.x,
							particle.y
							) < (sensor_range))
		{
			LandmarkObs nearby_landmark;
			nearby_landmark.id = map_landmarks.landmark_list[j].id_i;
			nearby_landmark.x = map_landmarks.landmark_list[j].x_f;
			nearby_landmark.y = map_landmarks.landmark_list[j].y_f;
			nearby_landmarks.push_back(nearby_landmark);	
		}
	}
	return nearby_landmarks;
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * Update the weights of each particle using a mult-variate Gaussian 
   */

	double weight_sum = 0.0f;
	for(int i=0; i<num_particles; ++i)
	{
		particles[i].weight = 1;	

		// transform observations to map coordinate system respect to a partice
		// outputs transformed_observations in map coordinates respect to the given particle
		vector<LandmarkObs> transformed_observations;
		for(unsigned int j=0; j<observations.size(); ++j)
		{
  		double x = particles[i].x + (cos(particles[i].theta) * observations[j].x) - 
					(sin(particles[i].theta) * observations[j].y);
  		double y = particles[i].y + (sin(particles[i].theta) * observations[j].x) + 
					(cos(particles[i].theta) * observations[j].y);

			LandmarkObs transformed_observation;
			transformed_observation.id = observations[j].id;
			transformed_observation.x = x;
			transformed_observation.y = y;
			transformed_observations.push_back(transformed_observation);
		}

		// Collect nearby landmarks 
	  // outputs nearby landmarks in map coordinates
		
		vector<LandmarkObs> nearby_landmarks;
		nearby_landmarks = filterLandmarks(particles[i], map_landmarks, sensor_range);
			
		// Create data associations between nearby landmarks and transformed obs
		// outputs transformed obs updated
		dataAssociation(nearby_landmarks, transformed_observations);

		// Calculate weights: go through nearby landmarks and calcuate weight
		// with associated observation
		for(unsigned int j=0; j<transformed_observations.size(); ++j)
		{
			for(unsigned int k=0; k<nearby_landmarks.size(); ++k)
			{
				if (transformed_observations[j].id == nearby_landmarks[k].id)
				{
					particles[i].weight = particles[i].weight * multiv_prob(std_landmark[0],
																std_landmark[1],
																transformed_observations[j].x, transformed_observations[j].y,
																nearby_landmarks[k].x, nearby_landmarks[k].y); 
					weight_sum+= particles[i].weight;
				}
			}
		}
	}

	// normalize weights
	for (int i=0;i<num_particles;++i)
	{
		particles[i].weight = particles[i].weight / weight_sum;		
		weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional 
   *   to their weight. 
   */
	std::default_random_engine gen;
	vector<Particle> resampled;
	std::uniform_int_distribution<int> index_dist(0, num_particles - 1);

	// resampling wheel variable
	double beta = 0.0f;

	// Select the largest weight
	double max_weight = *max_element(weights.begin(), weights.end());
	unsigned int curr_index = index_dist(gen);

	for(int i=0; i<num_particles; ++i)
	{
		std::uniform_real_distribution<double> dist_weight(0.0, 2 * max_weight);
		beta = beta + dist_weight(gen);
		// roll the resampling wheel
		while(beta > weights[curr_index])
		{
			beta = beta - weights[curr_index];
			curr_index = (curr_index + 1) % num_particles;
		}
		resampled.push_back(particles[curr_index]);
	}
	particles = resampled;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}


