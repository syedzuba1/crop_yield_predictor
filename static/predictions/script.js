document.addEventListener("DOMContentLoaded", () => {
    console.log("JavaScript file is loaded successfully!");

    // Select all tabs and graph containers
    const tabs = document.querySelectorAll(".tab");
    const graphs = document.querySelectorAll("#graph-content .graph");

    // Add event listeners to all tabs
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            // Remove "active" class from all tabs and hide all graphs
            tabs.forEach(t => t.classList.remove("active"));
            graphs.forEach(graph => graph.style.display = "none");

            // Add "active" class to clicked tab and show corresponding graph
            tab.classList.add("active");
            const selectedTab = tab.getAttribute("data-tab");
            document.getElementById(selectedTab).style.display = "block";
        });
    });

    // Ensure the default tab and graph are shown on page load
    const defaultTab = document.querySelector(".tab.active");
    if (defaultTab) {
        const defaultGraphId = defaultTab.getAttribute("data-tab");
        document.getElementById(defaultGraphId).style.display = "block";
    }
});
