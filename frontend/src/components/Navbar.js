import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import styles from '../assets/styles/Navbar.module.css';
import { RiArrowDropDownLine } from "react-icons/ri";

function Navbar({ showMenuSelect }) {
    const [dropdownOpen, setDropdownOpen] = useState(false);
    const [selectedOption, setSelectedOption] = useState("GPT4-o mini");

    const handleDropdownToggle = () => {
        setDropdownOpen((prev) => !prev);
    };

    const handleOptionSelect = (value) => {
        setSelectedOption(value);
        setDropdownOpen(false);
        console.log(`selected ${value}`);
    };

    return (
        <>
            <div className={styles.container}>
                <div className={styles.headers}>
                    <h1>RecLens</h1>
                </div>
                <div className={styles.menuButtons}>
                    <NavLink
                        to="/"
                        className={({ isActive }) =>
                            isActive ? `${styles.buttons} ${styles.active}` : styles.buttons
                        }
                    >
                        Home
                    </NavLink>
                    <NavLink
                        to="/recommend"
                        className={({ isActive }) =>
                            isActive ? `${styles.buttons} ${styles.active}` : styles.buttons
                        }
                    >
                        Recommend
                    </NavLink>
                    <NavLink
                        to="/profile"
                        className={({ isActive }) =>
                            isActive ? `${styles.buttons} ${styles.active}` : styles.buttons
                        }
                    >
                        Profile
                    </NavLink>
                </div>

                <div className={styles.menuSelect}>
                    {showMenuSelect && (
                        <div>
                            <button
                                className={styles.dropdownButton}
                                onClick={handleDropdownToggle}
                            >
                                {selectedOption}
                                <RiArrowDropDownLine />
                            </button>
                            {dropdownOpen && (
                                <div className={styles.dropdownMenu}>
                                    <div
                                        className={styles.dropdownItem}
                                        onClick={() => handleOptionSelect("GPT4-o mini")}
                                    >
                                        GPT4-o mini
                                    </div>
                                    <div
                                        className={styles.dropdownItem}
                                        onClick={() => handleOptionSelect("Gemini 1.5")}
                                    >
                                        Gemini 1.5
                                    </div>
                                    <div
                                        className={styles.dropdownItem}
                                        onClick={() => handleOptionSelect("Llama 3.2")}
                                    >
                                        Llama 3.2
                                    </div>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </>
    );
}

export default Navbar;
