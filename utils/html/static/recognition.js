setTimeout(checksocket, 3000);
var flag_speech = 0;
var flag_start = false;
recognitionFormControl(false);
var recognition = new webkitSpeechRecognition();

var rawText = "";
var confidenceLevel = -1;
var wSocket = null;

function asr_start_function() {
        flag_start = true;
        window.SpeechRecognition = window.SpeechRecognition || webkitSpeechRecognition;
        recognition = new webkitSpeechRecognition();
        var num = document.getElementById('selectLang').selectedIndex;
        var selectLang = document.getElementById('selectLang').options[num].value;
        var recognitionStartTime = null;
        //recognition.lang = 'ja';
        //console.log(selectLang);
        recognition.lang = selectLang;
        recognition.interimResults = true;
        recognition.continuous = true;
        
        recognition.onsoundstart = function() {
            document.getElementById('messageArea').innerHTML = "認識中";
            console.log("startrecog:");
        };
        recognition.onnomatch = function() {
            document.getElementById('messageArea').innerHTML = "もう一度試してください";
        };
        recognition.onerror = function() {
            document.getElementById('messageArea').innerHTML = "エラー";
            if(flag_speech == 0 && flag_start)
                asr_start_function();
            asr_stop_function();
            asr_start_function();
        };
        recognition.onsoundend = function() {
            document.getElementById('messageArea').innerHTML = "停止中";
            if(flag_start)asr_start_function();
        };
        recognition.onresult = function(event) {
            var results = event.results;
            
            for (var i = event.resultIndex; i < results.length; i++) {
                    if (results[i].isFinal)
                    {
                        rawText = results[i][0].transcript;
                        confidenceLevel = results[i][0].confidence;
                        if(wSocket != null){
                            if(recognitionStartTime==null){recognitionStartTime = new Date().toISOString();}
                            wSocket.send("result:" + rawText + "\nconfidence:" + confidenceLevel+ "\ntime_stamp:" + recognitionStartTime + "\n");
                            recognitionStartTime = null;
                        }
                        console.log( "result:"+rawText + "\nconfidence:" + confidenceLevel);
                        document.getElementById('result type').innerHTML = "[final]";
                        document.getElementById('confidence').innerHTML = confidenceLevel;
                        document.getElementById('recognitionText').innerHTML = results[i][0].transcript;
                        document.getElementById('recognitionText').classList.remove('isNotFinal');
                        // if(flag_start)asr_start_function();
                    }
                    else
                    {
                        rawText = results[i][0].transcript;
                        confidenceLevel = -1;
                        if(wSocket != null){
                            wSocket.send("in_progress");
                            recognitionStartTime = null;
                        }
                        if(wSocket != null){
                            wSocket.send("interimresult:"+rawText+"\n");
                        }
                        if(recognitionStartTime==null){recognitionStartTime = new Date().toISOString();}
                        console.log( "interimresult:"+rawText);
                        document.getElementById('result type').innerHTML = "[interim]";
                        document.getElementById('confidence').innerHTML = "-1";
                        document.getElementById('recognitionText').innerHTML = results[i][0].transcript;
                        document.getElementById('recognitionText').classList.add('isNotFinal');
                        flag_speech = 1;
                    }
            }
            //if(!flag_start)recognition.stop();
        }
        
        flag_speech = 0;
        document.getElementById('messageArea').innerHTML = "start";
        recognition.start();
        recognitionFormControl(true);
}

function asr_stop_function() {
    recognition.stop();
    document.getElementById('messageArea').innerHTML = "stop";
    flag_start = false;
    recognitionFormControl(false);
}

function recognitionFormControl(start){
    if(start){
        document.getElementById('recognitionStartButton').setAttribute('disabled','true');
        document.getElementById('recognitionStopButton').setAttribute('disabled','false');
        document.getElementById('recognitionStopButton').removeAttribute('disabled');
    }else{
        document.getElementById('recognitionStopButton').setAttribute('disabled','true');
        document.getElementById('recognitionStartButton').setAttribute('disabled','false');
        document.getElementById('recognitionStartButton').removeAttribute('disabled');
    }
}

function socketconnection(){
    var url = document.getElementById('socket_url').value;
    console.log("clicked "+url);
    if(wSocket == null){
        wSocket = new WebSocket(url);
        wSocket.onopen = function(event) {
            console.log("open socket "+event.data);
            document.getElementById('socket_connect_button').innerHTML = "<font size='2'>disconnect</font>";
            // if(!flag_start){
            // 	asr_start_function();
            // }
        };
        wSocket.onerror = function(error) {
            console.log(error.data);
        };
        wSocket.onmessage = function(event) {
            console.log(event.data);
            //publishData("socket="+event.data);
            if(event.data.indexOf('start') >= 0){
                if(flag_start){asr_stop_function();}
                asr_start_function();
            } else if(event.data.indexOf('stop') >= 0){
                if(flag_start){asr_stop_function();}
            }
        };
        wSocket.onclose = function() {
            console.log("closed socket");
            wSocket = null;
            document.getElementById('socket_connect_button').innerHTML = "<font size='2'>connect</font>";
        };
        console.log("connected to "+url);
    } else {
        if(wSocket.readyState == 1){
            wSocket.close();
            console.log("close socket");
        }
    }
}
function checksocket(){
    if(wSocket == null){
        socketconnection()
    }
}