
//텐서플로우 로드하기
//const tf = require('@tensorflow/tfjs');
//console.log('tensorflow 라이브러리 -->', tf);



// 1. 과거의 데이터를 준비합니다.
const temp = [12, 15, 10, 17];  //온도
const sold = [24, 30, 20, 34]; //판매량
const cause = tf.tensor(temp);  //원인 - 데이터를 tensor()함수로 자기들에 맞게 변형시킴.
const result = tf.tensor(sold); //결과

// 2. 모델의 모양을 만듭니다.
const x = tf.input({ shape: [1] });  //shape객체에 1을 지정하면 단 하나의 값이 들어옴을 의미한다.
const y = tf.layers.dense({ units: 1 }).apply(x);
let model = tf.model({ inputs: x, outputs: y });  //해당 model 데이터로 학습,예측 작업을 할것임
const compileParam = { optimizer: tf.train.adam(), loss: tf.losses.meanSquaredError }
//optimizer : 어떤 방법으로 컴파일할 것인지 선택해서 지정할 수 있다.
//loss : 만들어진 후, 다른 문제가 없는지 검사하는 방법
//meanSquaredError : 평균 제곱 오차
model.compile(compileParam);

// 3. 데이터로 모델을 학습시킵니다.
//let fitParam = { epochs: 1000 } //epochs :학습 횟수
let fitParam = {
    epochs: 1500, callbacks: {
        onEpochEnd: function (epoch, logs) {
            console.log('epoch', epoch, logs);  //  RMSE가 0 에가까울수록 학습이 잘 된것이라고 봄
            console.log('RMSE==>', Math.sqrt(logs.loss)); //logs.loss를 제곱근(sqrt)한 것
        }
    }
} // loss 추가 예제
model.fit(cause, result, fitParam).then(function (rs) {

    // 4. 모델을 이용합니다.
    // 4.1 기존의 데이터를 이용해서 예측한 데이터를 출력한다.
    let predict = model.predict(cause);
    predict.print();

});

                 // model.predict(tf.tensor([12])).print() 