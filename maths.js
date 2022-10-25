var answer;
var score = 0;
// var backgroundImages = [];
// var backgroundImages1 = [];

function nextQuestion(){
    const n1 = Math.floor(Math.random()*5);
    document.getElementById('n1').innerHTML = n1;

    const n2 = Math.floor(Math.random()*6);
    document.getElementById('n2').innerHTML = n2;

    answer = n1 + n2;
}

function checkAnswer() {
    const prediction = predictImage();
    console.log(`answer=${answer}, prediction=${prediction}`);
    if (prediction == answer){
        score++;
        console.log(`Correct!!. Score =${score}`);
        alert('Well Done!! Correct Answer');
        // backgroundImages.push(`url('images/background3.svg')`);
        // document.body.style.backgroundImage = backgroundImages;
    }
    if(prediction != answer){
            score--;
            console.log(`Wrong!!. Score =${score}`);
            alert('Oops!! Incorrect Answer');
            // backgroundImages1.push(`url('images/wrong_1.svg')`);
            // document.body.style.backgroundImage = backgroundImages1;   
    }
}


