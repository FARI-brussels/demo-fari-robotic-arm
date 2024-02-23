const video = document.querySelector('video');
const toggleSwitch = document.getElementById('toggleSwitch');
const reasoningDiv = document.getElementById('reasoning'); // Get the reasoning div
const planningDiv = document.getElementById('planning'); // Get the planning div

toggleSwitch.addEventListener('change', function () {
  if (this.checked) {
    // Call API endpoint
    fetch('http://localhost:8000/play', { method: 'POST' })
      .then(response => {
        if (response.ok) {
          return response.json(); // Parse the JSON of the response
        } else {
          throw new Error('Network response was not ok.');
        }
      })
      .then(data => {
        console.log("yu")
        // Update reasoning div with grid_state and next_move
        reasoningDiv.innerHTML = `<h3>Grid State:</h3><p>${data.reasoning.grid_state}</p><h3>Next Move:</h3><p>${data.reasoning.next_move}</p>`;

        // Display the base64-encoded image in the planning div
        const image = new Image(); // Create a new Image element
        image.src = 'data:image/jpeg;base64,' + data.frame; // Set the source to the base64-encoded image
        planningDiv.innerHTML = ''; // Clear the planning div before adding new content
        planningDiv.appendChild(image); // Add the image to the planning div
      })
      .catch(error => {
        console.error('There has been a problem with your fetch operation:', error);
      });
  } else {
    console.error('ya');
  }
});


// Example function to convert grid state string to HTML table
function gridStateToHTMLTable(gridState) {
  // Split the grid state into rows
  let rows = gridState.split('---------').map(row => row.trim().split('|'));

  // Start building the table HTML
  let tableHtml = '<table class="tic-tac-toe-grid">';

  rows.forEach(row => {
    tableHtml += '<tr>';
    row.forEach(cell => {
      tableHtml += `<td>${cell.trim() || '&nbsp;'}</td>`; // Use non-breaking space for empty cells
    });
    tableHtml += '</tr>';
  });

  tableHtml += '</table>';
  return tableHtml;
}

// Then, inside your fetch response handling:

