const video = document.querySelector('video');
const toggleSwitch = document.getElementById('toggleSwitch');
const reasoningDiv = document.getElementById('reasoning'); // Get the reasoning div

toggleSwitch.addEventListener('change', function () {
  if (this.checked) {
    // Call API endpoint
    fetch('http://localhost:8000/play', { method: 'POST' })
      .then(response => {
        if (response.ok) {
          return response.json(); // Parse the JSON of the response
        } else {
          throw new Error('Network response was not ok');
        }
      })
      .then(data => {
        // Update reasoning div with grid_state and next_move
        reasoningDiv.innerHTML = `<h3>Grid State:</h3><p>${data.reasoning.grid_state}</p><h3>Next Move:</h3><p>${data.reasoning.next_move}</p>`;
      })
      .catch(error => {
        console.error('There has been a problem with your fetch operation:', error);
      });
  } else {
    statusSpan.textContent = "Your Turn";
  }
});
