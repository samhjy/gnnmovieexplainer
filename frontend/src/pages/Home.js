import React from 'react';
import Navbar from '../components/Navbar';
import styles from '../assets/styles/Home.module.css';
import { NavLink } from 'react-router-dom';

function Home() {
    return (
        <>
        <div className={styles.pageContainer}>
            <Navbar showMenuSelect={false} />
            <div className={styles.container}>
                <div className={styles.textBox}>
                    <p className={styles.primary}>Movie Recommender</p>
                    <p className={styles.secondary}>Explainable AI.</p>
                    <p className={styles.desc}>
                        RecLens goes beyond basic recommendations, using cutting-edge AI to match you with films you’ll love and clearly explain why each pick fits your unique taste. Let's make every movie night smarter, together.
                    </p>
                    <div>
                        <NavLink className={styles.buttons} to="/recommend">Get Started</NavLink>
                    </div>
                </div>
            </div>
            </div>
        </>
    );
}

export default Home;