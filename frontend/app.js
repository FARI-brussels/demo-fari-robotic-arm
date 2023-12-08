const video = document.querySelector('video');
    const toggleSwitch = document.getElementById('toggleSwitch');

    toggleSwitch.addEventListener('change', function () {
      if (this.checked) {
        console.log("yooo")
        // Call API endpoint
        fetch('http://localhost:8000/play', { method: 'POST' })
          .then(response => {
            if (response.ok) {
              console.log("sbreeeeeeeeeee")
              this.checked = false;
              statusSpan.textContent = "Your Turn";
            } else {
              throw new Error('Network response was not ok');
            }
          })
          .catch(error => {
            console.error('There has been a problem with your fetch operation:', error);
          });
      } else {
        statusSpan.textContent = "Your Turn";
      }
    });
    function play() {
      fetch('/play', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        },
      })
      .then(response => response.json())
      .then(data => {
        console.log('Success:', data);
        if(data.status === 'success') {
          // Switch the slider back to "Your Turn"
          document.getElementById('slider').checked = false;
        }
      })
      .catch((error) => {
        console.error('Error:', error);
      });
    }
