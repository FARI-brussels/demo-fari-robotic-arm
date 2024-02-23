const video = document.querySelector('video');
    const toggleSwitch = document.getElementById('toggleSwitch');

    toggleSwitch.addEventListener('change', function () {
      if (this.checked) {
        // Call API endpoint
        fetch('http://localhost:8000/play', { method: 'POST' })
          .then(response => {
            if (response.ok) {
              console.log("re")
              console.log(response.json())
              //this.checked = false;
              //statusSpan.textContent = "Your Turn";
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
