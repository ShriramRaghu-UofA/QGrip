/** Browser-only dashboard bootstrap: mount the Svelte application at Vite's root element. */
import { mount } from 'svelte';
import App from './App.svelte';
import './style.css';

mount(App, { target: document.getElementById('app')! });
