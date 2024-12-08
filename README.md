# gnnmovieexplainer

## Environment Configuration
Navigate to the frontend folder and create a file named ```.env.development.local.``` Add your API key in the following format:
```
REACT_APP_OPENAI_API_KEY=your_openai_api_key_here
```

## Installation
### 1. Clone the Repository
Clone this repository to your local machine:
```
git clone https://github.com/your-username/gnnreccexplainer.git
cd gnnreccexplainer
```
### 2. Installing Required Packages
Navigate to the ```backend``` folder and install the required Python dependencies:
```
cd backend
pip install -r requirements.txt
```
Navigate to the ```frontend``` folder and install the required Node.js dependencies:
```
cd ../frontend
npm install
```
## Running the Application
From the root directory of the project, use the provided shell script to start both the backend and the frontend:
```
./run_gnnreccexplainer.sh
```
The backend will run on http://127.0.0.1:8000, and the frontend will be accessible at http://localhost:3000.
