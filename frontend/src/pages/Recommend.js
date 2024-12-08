import React, { useState } from 'react';
import Navbar from '../components/Navbar';
import { Slider } from 'antd';
import styles from '../assets/styles/Recommend.module.css';
import classNames from 'classnames';
import { getOpenAIResponse } from '../components/openaiService'; // Import the service

function Recommend() {
    const [sliderValue, setSliderValue] = useState(5);
    const [recommendations, setRecommendations] = useState([]);
    const [userMovies, setUserMovies] = useState([]);
    const [showRecommendations, setShowRecommendations] = useState(false);
    const [explanations, setExplanations] = useState(''); // State to store explanations

    const handleSliderChange = (value) => {
        setSliderValue(value);
    };

    const fetchRecommendations = async () => {
        try {
            const response = await fetch(`http://localhost:8000/lightgcn?user_id=1&num_recs=${sliderValue}`, {
                method: 'POST',
                headers: {
                    accept: 'application/json',
                },
            });
            const data = await response.json();
            console.log(data)
            setRecommendations(data.recommended_movies);
            setUserMovies(data.users_movies);
            setShowRecommendations(true);
        } catch (error) {
            console.error('Error fetching recommendations:', error);
        }
    };

    const generateExplanations = async () => {
        try {
            // Format the recommendations into a prompt string
            const recommendedMoviesList = recommendations
                .map((movie, index) => `${index + 1}. ${movie.title} - ${movie.genres}`)
                .join('\n');
            console.log(recommendedMoviesList)
            const likedMoviesList = userMovies
                .map((movie, index) => `${index + 1}. ${movie.title} - ${movie.genres}`)
                .join('\n');
            console.log(likedMoviesList)
            // Fetch the explanation from OpenAI
            const prompt = `
The following is an output generated with a trained LightGCN model:

Recommended Movies:
${recommendedMoviesList}

This recommendation is based on the following movies the user likes:
${likedMoviesList}

Explain to the user why each movie was recommended based on their preferences.
            `;

            const aiResponse = await getOpenAIResponse(prompt);
            setExplanations(aiResponse);

            setExplanations(aiResponse);
        } catch (error) {
            console.error('Error generating explanations:', error);
        }
    };

    return (
        <>
            <Navbar showMenuSelect={true} />
            <div
                className={classNames(
                    styles.container,
                    showRecommendations ? styles.recommendationsContainer : styles.sliderContainer
                )}
            >
                {showRecommendations ? (
                    <div className={styles.recommendationWrapper}>
                        <div className={styles.recommendationHeader}>
                            <img src={require('../assets/images/aiicon.png')} alt="AI Icon" className={styles.aiIcon} />
                            <p className={styles.desc}>
                                Based on your preferences, here are the top {sliderValue} movie recommendations that align well with your tastes:
                            </p>
                        </div>
                        <ul>
                            {recommendations.map((movie, index) => (
                                <li key={index} className={styles.listItem}>
                                    <strong>{movie.title}</strong> - {movie.genres}
                                </li>
                            ))}
                            <p>These films should resonate well with your interests! Enjoy watching!</p>
                        </ul>

                        {!explanations && (<button className={styles.explanationButton} onClick={generateExplanations}>
                            Generate Explanations
                        </button>)}
                        {explanations && (
                            <div className={styles.explanationHeader}>
                                <div className={styles.recommendationHeader}>
                                    <img src={require('../assets/images/aiicon.png')} alt="AI Icon" className={styles.aiIcon} />
                                    <p className={styles.desc}>
                                        {explanations ? explanations.split('\n')[0].trim() : ''}
                                    </p>
                                </div>
                                {explanations.split('\n').slice(1).map((line, index) => (
                                    <p className={styles.descexpl} key={index}>
                                    {line.replace(/[\*#]/g, '').trim()}
                                </p>
                                ))}
                            </div>
                        )}

                    </div>
                ) : (
                    <>
                        <p className={styles.primary}>Hello, Samantha</p>
                        <p className={styles.secondary}>
                            How many{' '}
                            <span className={styles.highlight}>movie recommendations</span> would you like today?
                        </p>
                        <Slider
                            className={styles.customSlider}
                            min={1}
                            max={10}
                            defaultValue={5}
                            onChange={handleSliderChange}
                        />
                        <button className={styles.buttons} onClick={fetchRecommendations}>
                            Get Recommendations
                        </button>
                    </>
                )}
            </div>
        </>
    );
}

export default Recommend;
